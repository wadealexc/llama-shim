import * as fs from 'fs';
import { PassThrough } from 'stream';
import path, { basename } from 'path';
import { spawn, ChildProcess } from 'child_process';
import fetch, { type RequestInit, Response } from 'node-fetch';

import express from 'express';
import chalk, { type ChalkInstance } from 'chalk';
import bytes from 'bytes';

import * as utils from './utils.js';
import { LlamaManager } from './llamaManager.js';

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
const HOST_IP = utils.getLanIPv4();                   // Our ip address
const LLAMA_SERVER_URL = `http://${HOST_IP}:${INTERNAL_PORT}`;

const CONFIG_PATH = './shim-config.json';
const CONFIG_JSON = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
const DEFAULT_MODEL: string = CONFIG_JSON['default'];
const MODEL_FILES_PATH: string = CONFIG_JSON['models'];
const LOG_DIR: string = CONFIG_JSON['logs'];

// Find all models recursively in MODEL_FILES_PATH and check for the default model
const MODELS = utils.findModels(MODEL_FILES_PATH);
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

let currentModelPath = DEFAULT_MODEL_PATH;
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const logFilePrefix = `llama-${timestamp}`

const chatPath = path.join(LOG_DIR, `${logFilePrefix}.chat.json`);
const chatLogs: any[] = [];
const chatStream: fs.WriteStream = fs.createWriteStream(chatPath, { flags: 'a' });

/* -------------------- INIT LLAMA AND EXPRESS -------------------- */

// Start llama-server
const llama = new LlamaManager(HOST_IP, INTERNAL_PORT, DEFAULT_MODEL_PATH, LOG_DIR, logFilePrefix);

const app = express();
app.use(express.raw({ type: (() => true), limit: '50mb' }));

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
    const targetModelPath = MODELS.find(p => utils.modelNameFromPath(p) === model);
    if (!targetModelPath) {
        console.error(`Requested model "${model}" not found locally; ignoring`);
        return currentModelPath;
    }

    return targetModelPath;
}

/* -------------------- ROUTES -------------------- */

/**
 * /v1/models – return list of local models
 */
app.get('/v1/models', async (_req, res) => {
    try {
        const data = MODELS.map(p => ({
            id: utils.modelNameFromPath(p),
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

    let llamaResponse: Response | null = null;
    try {
        // Ensure the correct model is running before forwarding
        const modelPath = parseModelPath(body.model as string | undefined);
        if (modelPath !== currentModelPath) {
            console.log(`Requested new model ${utils.modelNameFromPath(modelPath)}; restarting llama-server`);
            try { await llama.restartServer(modelPath) } catch (err) {
                console.error(`error restarting llama-server: `, err);
                await shutdown('SIGTERM');
                return;
            }

            currentModelPath = modelPath;
        }

        // Forward request to llama-server. If llama is busy, we'll wait here.
        llamaResponse = await llama.forwardRequest({
            originalURL: req.originalUrl,
            headers: req.headers,
            method: req.method,
            body: req.body
        });

        // copy headers and status from upstream
        res.status(llamaResponse.status);
        llamaResponse.headers.forEach((v, k) => res.setHeader(k, v));

        passThrough.once('error', err => {
            console.error('passThrough error', err);
            res.destroy(err);
        });

        llamaResponse.body?.once('error', err => {
            console.error('upstream body error', err);
            passThrough.destroy(err);
        });

        // Stream llama-server response to both the client as well as our logs
        llamaResponse.body?.pipe(passThrough).pipe(res);
    } catch (err: any) {
        console.error('Proxy error:', err);
        res.status(502).json({ error: err.message || 'Bad gateway' });
    } finally {
        const len = req.headers['content-length'] ?? '0';

        console.log(`${req.method}: ${chalk.yellow(req.originalUrl)} (req len: ${bytes(Number(len))})`);
        if (!llamaResponse) {
            console.log(chalk.dim.red(` -> failed before request made to llama-server`));
        } else if (llamaResponse.status === 200) {
            console.log(chalk.dim.green(` -> llama-server responds success (200)`));
        } else {
            console.log(chalk.dim.yellow(` -> llama-server responds with (${llamaResponse.status})`));
        }
    }
});

/* -------------------- START EXPRESS -------------------- */

app.listen(EXTERNAL_PORT, () => {
    console.log(`llama-shim listening on ${chalk.cyan(`http://${HOST_IP}:${EXTERNAL_PORT}`)}`);
    console.log(chalk.dim(` - proxying to internal llama‑server on: ${chalk.cyan(INTERNAL_PORT)}`));
    console.log(`Local GGUF models: [\n${MODELS.map(p => chalk.magenta(utils.modelNameFromPath(p))).join(',\n')}\n]`);
    console.log(`Using default model: ${chalk.magenta(DEFAULT_MODEL)}`);
});

/* -------------------- STOP SERVER -------------------- */

async function shutdown(signal: string) {
    console.log(`shutting down llama-shim (${chalk.yellow(signal)})`);

    await Promise.allSettled([
        // 1. Kill llama-server if it's running
        new Promise<void>(async (resolve) => {
            try { await llama.stopServer() }
            catch (err) { console.error(`shutdown: llama.stopServer failed: `, err) }
            finally { resolve() }
        }),
        // 2. Write chat logs to file
        new Promise<void>(async (resolve) => {
            chatStream.once('finish', resolve);
            chatStream.once('error', (err) => {
                console.error(`shutdown: error writing logs to file: `, err);
                resolve();
            });

            // Actually do the write, then close the file
            chatStream.write(JSON.stringify(chatLogs, null, 2));
            chatStream.end();
        })
    ]);

    console.log('... done!');
}

// Listen to OS signals, shutdown llama if needed
for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.once(signal, async () => {
        await shutdown(signal);
        process.exit(0);
    });
}

// Shutdown if we have any uncaught errors
for (const evt of ['uncaughtException', 'unhandledRejection'] as const) {
    process.once(evt, async (err) => {
        console.error(evt, err);
        await shutdown('SIGTERM');
        process.exit(1);
    });
}

process.once('exit', () => {
    // Attempt last-ditch cleanup, sending SIGKILL if we still have a process running
    try { llama.forceStopServer() } catch { };
});