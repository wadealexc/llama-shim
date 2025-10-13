import * as fs from 'fs';
import { PassThrough } from 'stream';
import path, { basename } from 'path';
import { networkInterfaces } from "os";
import { spawn, ChildProcess } from 'child_process';
import fetch, { type RequestInit, Response } from 'node-fetch';

import express from 'express';
import chalk from 'chalk';

type ChatResponse = {
    created: number,
    id: string,
    model: string,
    system_fingerprint: string,
    object: string,
    timings?: any,
    choices: ChatChoice[],
};

type ChatChoice = {
    finish_reason: string | null,
    index: number,
    delta: {
        content?: string,
        reasoning_content?: string,
    }
};

/* -------------------- CONFIGURATION -------------------- */
const EXTERNAL_PORT = 8081;                     // Port the proxy listens on
const INTERNAL_PORT = 8080;                     // Port llama‑server runs on
const HOST_IP = getLanIPv4();                   // Our ip address
const LLAMA_SERVER_URL = `http://${HOST_IP}:${INTERNAL_PORT}`;

const MODEL_FILE_EXT = '.gguf';                 // File extension for local models
const LLM_COMMAND = 'llama-server';             // Command to start llama‑server

const CONFIG_PATH = './shim-config.json';
const CONFIG_JSON = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
const DEFAULT_MODEL: string = CONFIG_JSON['default'];
const MODEL_FILES_PATH: string = CONFIG_JSON['models'];
const LOG_DIR: string = CONFIG_JSON['logs'];

// Find all models recursively in MODEL_FILES_PATH and check for the default model
const MODELS = findModels(MODEL_FILES_PATH);
if (MODELS.length === 0) {
    console.error('No local GGUF models found in the working directory.');
    process.exit(1);
}

const DEFAULT_MODEL_PATH = MODELS.find(model => basename(model) === DEFAULT_MODEL)
    ?? (() => { throw new Error(`found ${MODELS.length} models; did not find requested default model: ${DEFAULT_MODEL}`); })();

// ensure log directory exists
try { fs.mkdirSync(LOG_DIR, { recursive: true }); } catch (e) {
    console.error('Failed to create log dir', e);
    process.exit(1);
}

/* -------------------- UTILS -------------------- */

// Recursively find all .gguf files in a given directory
function findModels(dir: string): string[] {
    let results: string[] = [];

    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
            // Recurse into subdirectories
            results = results.concat(findModels(fullPath));
        } else if (entry.isFile() && fullPath.endsWith(".gguf")) {
            // Match .gguf files
            results.push(fullPath);
        }
    }

    return results;
}

/**
 * Strip the .gguf extension from a model file
 * (input: "gpt-oss-20b-mxfp4.gguf", output: "gpt-oss-20b-mxfp4")
 */
function modelNameFromPath(p: string): string {
    return path.basename(p, MODEL_FILE_EXT);
}

function getLanIPv4(): string {
    const nets = networkInterfaces();

    for (const name of Object.keys(nets)) {
        for (const net of nets[name] ?? []) {
            // skip over non‑IPv4 and internal (loopback) addresses
            if (net.family === "IPv4" && !net.internal) {
                return net.address; // e.g., "192.168.1.42"
            }
        }
    }

    throw new Error(`unable to find valid LAN IPv4 address`);
}

/* -------------------- LLAMA-SERVER CONTROLLER -------------------- */

let currentModelPath = DEFAULT_MODEL_PATH;
let llamaProcess: ChildProcess | null = null;
let llamaAbort: AbortController | null = null;
let llamaStarting: Promise<void> | null = null;

const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = path.join(LOG_DIR, `llama-${timestamp}.out.log`);
const errPath = path.join(LOG_DIR, `llama-${timestamp}.err.log`);
const chatPath = path.join(LOG_DIR, `llama-${timestamp}.chat.json`);

// open append write streams for stdout/stderr
const stdoutStream: fs.WriteStream = fs.createWriteStream(outPath, { flags: 'a' });
const stderrStream: fs.WriteStream = fs.createWriteStream(errPath, { flags: 'a' });

const chatLogs: any[] = [];
const chatStream: fs.WriteStream = fs.createWriteStream(chatPath, { flags: 'a' });

/**
 * Start the llama‑server with the given model path.
 */
async function startLlamaServer(modelPath: string) {
    // Don't start llama-server if we already have an active child process
    if (llamaProcess) throw new Error(`Unable to stop llama-server before timeout elapsed`);

    // If we're currently starting up, reuse that promise
    if (llamaStarting) return llamaStarting;

    llamaAbort = new AbortController();

    llamaStarting = (async () => {
        console.log(`Starting llama-server with model: ${chalk.magenta(modelPath)}`);
        const args = [
            '-m', modelPath,
            '--host', HOST_IP,
            '--port', INTERNAL_PORT.toString(),
            '--ctx-size', '0',
            '--jinja',
            '-fa', '1',
        ];

        // spawn llama-server
        llamaProcess = spawn(LLM_COMMAND, args, {
            stdio: ['ignore', 'pipe', 'pipe'],
            signal: llamaAbort.signal,
        });

        // pipe child's stdout/stderr to files
        if (llamaProcess.stdout && stdoutStream) llamaProcess.stdout.pipe(stdoutStream);
        if (llamaProcess.stderr && stderrStream) llamaProcess.stderr.pipe(stderrStream);

        llamaProcess.once('error', (err) => {
            // Don't log AbortErrors; we get these when we kill the process
            if ((err as DOMException)?.name === 'AbortError') {
                console.log(`llama-server received abort`);
            } else {
                console.error('llama-server process error:', err);
            }

            llamaAbort?.abort();
        });

        llamaProcess.once('close', (code, signal) => {
            console.log(`llama-server exited code=${code} signal=${signal}`);
            llamaAbort?.abort();
        });

        process.stdout.write(chalk.dim('waiting for llama-server to load model... '));
        // const startTs = Date.now();

        // Polling loop - try to query llama-server until success, abort or timeout
        while (true) {
            if (llamaAbort.signal.aborted) {
                throw new Error(`Aborted while waiting for llama-server to start up`);
            }

            // Create per-request abort; link to main abort
            const reqAbort = new AbortController();
            const onOuterAbort = () => reqAbort.abort();
            llamaAbort.signal.addEventListener('abort', onOuterAbort);

            // Super short timeout for each request since it's all on this machine
            const reqTimeout = setTimeout(() => reqAbort.abort(), 100);
            reqTimeout.unref();

            try {
                process.stdout.write(chalk.dim(`ping...`));
                const res = await fetch(LLAMA_SERVER_URL, { signal: reqAbort.signal });

                if (res.ok) {
                    process.stdout.write(chalk.green('done!\n'));

                    // Log and return - ready for requests!
                    currentModelPath = modelPath;
                    console.log(chalk.green(`llama-server is now running on port ${INTERNAL_PORT}`));
                    console.log(chalk.dim(` - logs -> stdout: ${outPath}`));
                    console.log(chalk.dim(` - logs -> stderr: ${errPath}`));
                    return;
                }
            } catch (err) {
                // swallow err; keep polling
            } finally {
                // Clean up timeout/request signal
                clearTimeout(reqTimeout);
                llamaAbort.signal.removeEventListener('abort', onOuterAbort);
            }

            // sleep half a second, then retry
            await new Promise((r) => setTimeout(r, 500)).then(() => undefined);
        }
    })();

    try {
        await llamaStarting;       // ensure this call doesn't return until ready
    } finally {
        llamaStarting = null;      // clear the lock for future starts
    }
}

async function stopLlamaServer() {
    if (!llamaProcess) return;

    const proc = llamaProcess;
    console.log(chalk.dim(`killing existing llama-server (pid ${proc.pid})... `));

    llamaProcess = null;
    llamaAbort?.abort();

    const done = new Promise<void>((resolve) => {
        proc.once('close', () => resolve());
        proc.once('exit', () => resolve());
    });

    // Wait for llama-server to exit cleanly before returning
    await done;

    console.log(chalk.green(`done!`));
}

/**
 * Determine the model to use for an incoming request.
 * 
 * If the request specifies a model that is different from the one
 * currently running, return the path of the model that should be run.
 * If we're already running the correct model, return the path of the
 * already-running model.
 */
function parseModelPath(model: string | undefined): string {
    if (!model) {
        console.error(`Request did not specify model; ignoring`);
        return currentModelPath;
    }

    // Resolve the target model path from the list of local models
    const targetModelPath = MODELS.find(p => modelNameFromPath(p) === model);
    if (!targetModelPath) {
        console.error(`Requested model "${model}" not found locally; ignoring`);
        return currentModelPath;
    }

    return targetModelPath;
}

/* -------------------- EXPRESS SETUP -------------------- */

const app = express();
app.use(express.raw({ type: () => true }));

/**
 * /v1/models – return list of local models
 */
app.get('/v1/models', async (_req, res) => {
    try {
        const data = MODELS.map(p => ({
            id: modelNameFromPath(p),
            object: 'model',
            owned_by: 'llamacpp',
        }));
        res.json({ object: 'list', data });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: 'Failed to list models' });
    }
});

/**
 * Proxy all other routes to the internal llama‑server
 */
app.all('/{*splat}', async (req, res) => {
    let body = req.body || {};

    // request: X / response: X will be logged
    let chatLog: {
        request?: any,
        response?: any,
    } = {};

    // Convert request body to JSON
    if (Buffer.isBuffer(body) && req.headers['content-type']?.includes('application/json')) {
        try {
            const text = body.toString('utf8');
            body = JSON.parse(text);
            chatLog.request = body.messages as {
                content: string | any[],
                role: string
            }[] | undefined;

            if (!chatLog.request) {
                console.error(`body.messages used unexpected type: ${body.messages}`);
            }
        } catch (e) {
            console.warn('Failed to parse JSON body:', e);
        }
    }

    // As the response comes back from llama-server, we'll collect it
    // here so we can log it without getting in the way of the shim response.
    //
    // The response is formatted as SSE (server-sent events).
    // https://platform.openai.com/docs/api-reference/chat-streaming
    const passThrough = new PassThrough();
    const chunks: Buffer<any>[] = [];

    passThrough.on('data', chunk => chunks.push(Buffer.from(chunk)));
    passThrough.on('end', () => {
        const body = Buffer.concat(chunks).toString('utf8');

        // Try to decode as JSON
        try {
            chatLog.response = JSON.parse(body);
            chatLogs.push(chatLog);
            return;
        } catch (e) {
            // This won't be JSON if the request was made with `stream: true`
            // Swallow error, handle below
        }

        // Parse SSE 'data' events
        if (!chatLog.response) {
            // This should give us an array of JSON strings representing various streamed tokens
            chatLog.response = body.split('data: ').reduce<ChatResponse[]>((accum, event) => {
                try {
                    accum.push(JSON.parse(event.trim()));
                } catch {

                }

                return accum;
            }, []);
        }

        if (!chatLog.response) {
            console.error('failed to decode llama-server response');
            return;
        }

        // Combine streamed ChatResponse deltas into a single ChatResponse for logs
        let finalResponse: ChatResponse | null = null;
        let finalContent: string = '';
        let finalReasoningContent: string = '';

        (chatLog.response as ChatResponse[]).forEach(response => {
            const choice = response.choices.at(0) as ChatChoice;

            if (choice.delta.content) {
                finalContent += choice.delta.content;
            }

            if (choice.delta.reasoning_content) {
                finalReasoningContent += choice.delta.reasoning_content;
            }

            finalResponse = {
                ...response,
                choices: [{
                    ...choice,
                    delta: {
                        content: finalContent,
                        reasoning_content: finalReasoningContent,
                    }
                }]
            }    
        })

        if (!finalResponse) {
            console.error(`llama-server response in unknown format`);
            return;
        }

        chatLog.response = finalResponse;
        chatLogs.push(chatLog);
    });

    try {
        // If the server is currently starting up, wait for it to finish before processing a request
        if (llamaStarting) await llamaStarting;

        // Ensure the correct model is running before forwarding
        const modelPath = parseModelPath(body.model as string | undefined);
        if (modelPath !== currentModelPath) {
            console.log(`Requested new model ${modelNameFromPath(modelPath)}; restarting llama-server`);
            await stopLlamaServer();
            await startLlamaServer(modelPath);
        }

        // Log chat messages
        // const chatLog = body.messages as {}[] | undefined;

        // Build the URL for `llama-server`
        const llamaURL = `${LLAMA_SERVER_URL}${req.originalUrl}`;

        // Sanitize headers
        const headers: HeadersInit = {};
        for (const [key, value] of Object.entries(req.headers)) {
            if (typeof value === "string") headers[key] = value;
            else if (Array.isArray(value)) headers[key] = value.join(", ");
        }

        // ensure Host header matches target
        headers["host"] = LLAMA_SERVER_URL;

        const init: RequestInit = {
            method: req.method,
            headers,
            body: req.method !== "GET" && req.method !== "HEAD" ? req.body : undefined,
        };

        const upstream = await fetch(llamaURL, init);

        // copy headers and status from upstream
        res.status(upstream.status);
        upstream.headers.forEach((v, k) => res.setHeader(k, v));

        passThrough.once('error', err => {
            console.error('passThrough error', err);
            res.destroy(err);
        });

        upstream.body?.once('error', err => {
            console.error('upstream body error', err);
            passThrough.destroy(err);
        });

        // Stream llama-server response to both the client as well as our logs
        upstream.body?.pipe(passThrough).pipe(res);
    } catch (err: any) {
        console.error('Proxy error:', err);
        res.status(502).json({ error: err.message || 'Bad gateway' });
    }
});

/* -------------------- START SERVER -------------------- */

try {
    await startLlamaServer(DEFAULT_MODEL_PATH);
} catch (err) {
    console.error('Failed to start llama-server:', err);
    process.exit(1);
}

app.listen(EXTERNAL_PORT, () => {
    console.log(`llama-shim listening on ${chalk.cyan(`http://${HOST_IP}:${EXTERNAL_PORT}`)}`);
    console.log(chalk.dim(` - proxying to internal llama‑server on: ${chalk.cyan(INTERNAL_PORT)}`));
    console.log(`Local GGUF models: [\n${MODELS.map(p => chalk.magenta(modelNameFromPath(p))).join(',\n')}\n]`);
    console.log(`Using default model: ${chalk.magenta(DEFAULT_MODEL)}`);
});

/* -------------------- STOP SERVER -------------------- */

async function shutdown(signal: string) {
    console.log(`Shutting down llama-shim (${chalk.yellow(signal)})`);

    // 1. Write chat logs to file
    chatStream.write(JSON.stringify(chatLogs, null, 2));

    // 2. Kill llama-server if it's running
    await stopLlamaServer();

    // 3. Close log streams
    try {
        stdoutStream.end();
        stderrStream.end();
        chatStream.end();
    } catch (err) {
        console.error(`Unable to end output streams: ${err}`);
    }
}

// Listen to OS signals
process.once("SIGINT", () => shutdown("SIGINT").then(() => process.exit(0)));
process.once("SIGTERM", () => shutdown("SIGTERM").then(() => process.exit(0)));

// On uncaught exceptions / rejections, attempt to shutdown then exit nonzero
process.on("uncaughtException", (err) => {
    console.error("uncaughtException:", err);
    shutdown("uncaughtException").then(() => process.exit(1));
});
process.on("unhandledRejection", (reason) => {
    console.error("unhandledRejection:", reason);
    shutdown("unhandledRejection").then(() => process.exit(1));
});