import * as fs from 'fs';
import path, { basename } from 'path';
import { networkInterfaces } from "os";
import { spawn, ChildProcess } from 'child_process';
import fetch, { type RequestInit, Response } from 'node-fetch';


import express from 'express';
import chalk from 'chalk';

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

// open append write streams for stdout/stderr
let stdoutStream: fs.WriteStream = fs.createWriteStream(outPath, { flags: 'a' });
let stderrStream: fs.WriteStream = fs.createWriteStream(errPath, { flags: 'a' });

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
            console.error('llama-server process error:', err);
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
                    console.log(`llama-server is now running on port ${INTERNAL_PORT}`);
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
    llamaProcess = null;
    llamaAbort?.abort();

    const done = new Promise<void>((resolve) => {
        proc.once('close', () => resolve());
        proc.once('exit', () => resolve());
    });

    process.stdout.write(chalk.dim(`killing existing llama-server (pid ${proc.pid})... `));

    // Wait for llama-server to exit cleanly before returning
    try { proc.kill('SIGTERM'); } catch { }
    await done;

    process.stdout.write(chalk.green(`done!\n`));
}

/**
 * Determine the model to use for an incoming request.
 * 
 * If the request specifies a model that is different from the one
 * currently running, return the path of the model that should be run.
 * If we're already running the correct model, return the path of the
 * already-running model.
 */
function parseModelPath(req: express.Request): string {
    let body = req.body || {};

    // If it's a Buffer and content type is JSON, parse it
    if (Buffer.isBuffer(body) && req.headers['content-type']?.includes('application/json')) {
        try {
            const text = body.toString('utf8');
            body = JSON.parse(text);
            console.log(JSON.stringify(body, null, 2));
        } catch (e) {
            console.warn('Failed to parse JSON body:', e);
        }
    }

    const requestedModel = body.model as string | undefined;

    if (!requestedModel) {
        console.error(`Request did not specify model; ignoring`);
        return currentModelPath;
    }

    // Resolve the target model path from the list of local models
    const targetModelPath = MODELS.find(p => modelNameFromPath(p) === requestedModel);
    if (!targetModelPath) {
        console.error(`Requested model "${requestedModel}" not found locally; ignoring`);
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
    try {
        // If the server is currently starting up, wait for it to finish before processing a request
        if (llamaStarting) await llamaStarting;

        // Ensure the correct model is running before forwarding
        let modelPath = parseModelPath(req);
        if (modelPath !== currentModelPath) {
            console.log(`Requested new model ${modelNameFromPath(modelPath)}; restarting llama-server`);
            await stopLlamaServer();
            await startLlamaServer(modelPath);
        }

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

        // stream the response directly back to the client
        upstream.body?.pipe(res);
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

    // 1. Kill llama-server if it's running
    await stopLlamaServer();

    // 2. Close log streams
    try {
        stdoutStream.end();
        stderrStream.end();
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