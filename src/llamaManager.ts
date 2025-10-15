import path from 'path';
import * as fs from 'node:fs';
import { spawn, ChildProcess } from 'child_process';
import fetch, { type RequestInit, Response } from 'node-fetch';

import chalk from 'chalk';

import * as utils from './utils.js';
import type { IncomingHttpHeaders } from 'node:http';

// Command to start llama‑server
//
// In order for this to work, `llama-server` needs to be in your $PATH
const LLM_COMMAND = 'llama-server';

// How long to wait for a graceful shutdown before sending SIGKILL
const SHUTDOWN_GRACE_MS = 5000;

// How long to wait for a poll to succeed
// (we shouldn't need long since this is all on the same machine)
const POLL_TIMEOUT_MS = 100;

// Frequency of pings sent to `llama-server` on startup
const POLL_INTERVAL_MS = 500;

// Number of times to ping `llama-server` process before giving up on life
// (total wait time ms: NUM_RETRIES * POLL_INTERVAL_MS -> 20 seconds)
const NUM_RETRIES = 40;

export class LlamaManager {

    private llamaServerIP: string;
    private llamaServerPort: string;
    private llamaServerURL: string;

    private llamaProcess: ChildProcess | null = null;
    private llamaExited: Promise<void> | null = null;

    private runCounter = 0;
    private activeRequests = 0;
    private restartInProgress: Promise<void> | null = null;

    private logDirectory: string;
    private logFilePrefix: string;

    constructor(
        llamaServerIP: string, llamaServerPort: number,
        defaultModelPath: string,
        logDirectory: string, logFilePrefix: string,
    ) {
        this.llamaServerIP = llamaServerIP;
        this.llamaServerPort = llamaServerPort.toString();
        this.llamaServerURL = `http://${llamaServerIP}:${llamaServerPort}`;

        this.logDirectory = logDirectory;
        this.logFilePrefix = logFilePrefix;

        this.restartServer(defaultModelPath);
    }

    /* -------------------- PUBLIC METHODS -------------------- */

    /**
     * Kill the existing llama-server process and start a new one, loading the GGUF
     * model given by `modelPath`. This can take some time, as the old server
     * needs to be shut down, and the new model needs to be loaded.
     * 
     * This method will throw if we're already working on a restart, or if we have an
     * active `forwardRequest`.
     * 
     * @param modelPath Path to the GGUF file to load into llama-server
     */
    async restartServer(modelPath: string) {
        if (this.restartInProgress) throw new Error(`LlamaManager.restartServer: attempted restart while restart in progress`);
        if (this.activeRequests !== 0) throw new Error(`LlamaManager.restartServer: attempted restart while requests active`);

        // Create a promise for the restart process that resolves when restart is complete
        this.runCounter++;
        this.restartInProgress = new Promise<void>(async (resolve) => {
            try {
                await this.stopServer();
                await this.#startServer(modelPath);
                await this.#pollServer();
            } finally {
                this.restartInProgress = null;
                resolve();
            }
        });
    }

    /**
     * Kill the llama-server process, if it's running. Attempts a graceful shutdown,
     * using SIGTERM and waiting `SHUTDOWN_GRACE_MS`. If this does not succeed, this
     * method sends SIGKILL and waits `2 * SHUTDOWN_GRACE_MS`.
     * 
     * If the process is still running, throws an error.
     */
    async stopServer() {
        if (!this.llamaProcess) return;
        const pid = this.llamaProcess.pid!;

        // Shutdown handler that sends a kill signal to the process group, then waits
        // for a grace period. Resolves with `true` if shutdown occurred, `false` if
        // the grace period expired.
        const shutdownWithgracePeriod = (async (signal: string, pid: number, gracePeriodMS: number) => {
            try { process.kill(-pid, signal) } catch { };

            return Promise.race([
                new Promise<boolean>(async (resolve) => {
                    if (this.llamaExited) await this.llamaExited;

                    resolve(true);
                }),
                new Promise<boolean>((resolve) => {
                    setTimeout(() => resolve(false), gracePeriodMS);
                })
            ])
        });

        // try a graceful shutdown first (SIGTERM)
        console.log(chalk.dim(`killing llama-server process (pid ${pid}) (${chalk.yellow('SIGTERM')})...`));
        if (await shutdownWithgracePeriod('SIGTERM', pid, SHUTDOWN_GRACE_MS)) {
            console.log(chalk.green('done! (graceful shutdown)'));
            return;
        }

        // grace period's up, now it's business (SIGKILL)
        //
        // *teleports behind you* "nothin personnel, kid"
        console.log(`failed to stop process gracefully, sending ${chalk.yellow('SIGKILL')}...`);
        if (await shutdownWithgracePeriod('SIGKILL', pid, 2 * SHUTDOWN_GRACE_MS)) {
            console.log(chalk.yellow(`done! (forced shutdown)`));
            return;
        }

        // if we don't get a shutdown, burn it all to the ground
        throw new Error(`failed to kill llama-server (pid ${pid})`);
    }

    /**
     * Immediately SIGKILL any llama-server process (or its children), if they exist
     * Returns without checking to see if the process ended.
     */
    forceStopServer() {
        if (!this.llamaProcess) return;
        const pid = this.llamaProcess.pid!;

        // we need an immediate exit, show no mercy
        console.log(chalk.dim(`killing llama-server process (pid ${pid}) (${chalk.yellow('SIGKILL')})...`));
        try { process.kill(-pid, 'SIGKILL') } catch { };
    }

    async forwardRequest(req: {
        originalURL: string,
        headers: IncomingHttpHeaders,
        method: string,
        body: any,
    }): Promise<Response> {
        // If the llama-server process is being restarted, wait for it to complete before continuing
        if (this.restartInProgress) {
            console.log(`LlamaManager.forwardRequest: restart in progress, waiting...`);
            try { await this.restartInProgress }
            catch (err) { console.error(`LlamaManager.forwardRequest: restart failed: ${err}, continuing`) }            
        }

        if (!this.llamaProcess) throw new Error(`LlamaManager.forwardRequest: no llama process found!`);

        // Increment active requests, preventing calls to `restartServer` until this request is processed
        // Note: this does NOT prevent calls to `stopServer` or `forceStopServer`, as we don't want to
        // block shutdown.
        this.activeRequests++;

        try {
            const llamaURL = this.llamaServerURL + req.originalURL;

            // Convert IncomingHttpHeaders to HeadersInit
            const headers: HeadersInit = {};
            for (const [key, value] of Object.entries(req.headers)) {
                if (typeof value === 'string') headers[key] = value;
                else if (Array.isArray(value)) headers[key] = value.join(', ');
            }

            // ensure Host header matches target
            headers["host"] = this.llamaServerURL;

            // Construct request
            const init: RequestInit = {
                method: req.method,
                headers,
                body: !(req.method === 'GET' || req.method === 'HEAD') ? req.body : undefined,
            };

            return fetch(llamaURL, init);
        } finally {
            this.activeRequests--;
        }
    }

    /* -------------------- PRIVATE METHODS -------------------- */

    /**
     * Start a new `llama-server` process, loading the GGUF file located at `modelPath`.
     * Opens a logfile for the process, and starts `llama-server` as a detached process
     * in its own process group.
     * 
     * Also sets up this.llamaExited, which resolves when the process exits.
     * 
     * @param modelPath Path to the GGUF file to load into llama-server
     */
    async #startServer(modelPath: string) {
        if (this.llamaProcess) throw new Error(`LlamaManager.startServer: llama process already running`);

        // Open log file for `llama-server` stdout/stderr
        // File name example: `llama-{timestamp}_r0_gpt-oss-20b.log`
        const logFileName =
            this.logFilePrefix +
            '_r' + this.runCounter.toString() +
            '_' + utils.modelNameFromPath(modelPath) + '.log';
        const logPath = path.join(this.logDirectory, logFileName);

        // Open log file for child process stdout/stderr
        console.log(`starting llama-server with model: ${chalk.magenta(modelPath)} | ${chalk.dim(`(${modelPath})`)}`);
        console.log(chalk.dim(` - using log file: ${logPath}`));
        const out = fs.openSync(logPath, 'a');
        const err = fs.openSync(logPath, 'a');

        const args = [
            '-m', modelPath,
            '--host', this.llamaServerIP,
            '--port', this.llamaServerPort,
            '--ctx-size', '0',
            '--jinja',
            '-fa', '1',
        ];

        // Spawn llama-server as a detached process, making it the leader of a new
        // process group. This means that we can send a kill signal to `-pid` to also
        // send a signal to any child processes it spawns.
        const proc = spawn(LLM_COMMAND, args, {
            stdio: ['ignore', out, err],
            detached: true,
        });

        // 'error' is emitted when:
        // - process could not be spawned
        // - process could not be killed
        // - sending a message/signal to the process failed
        // - process was aborted via signal
        //
        // ... since this handles a lot of scenarios, we just log here.
        proc.once('error', (err) => {
            console.error('llama-server process error:', err);
        });

        // (we use `proc` to bind callbacks to this context)
        this.llamaProcess = proc;

        // Create a promise that resolves when llamaProcess exits
        //
        // 'close' will only be emitted once the process exits AND all stdio streams
        // have been closed. 'exit' will be emitted once the process exits.
        this.llamaExited = new Promise<void>((resolve) => {
            // proc.once('close', () => resolve());
            proc.once('exit', (code, signal) => {
                console.log(`llama-server exited code=${code} signal=${signal}`);

                // clean up log files
                try { fs.closeSync(out) } catch { };
                try { fs.closeSync(err) } catch { };

                this.llamaProcess = null;
                this.llamaExited = null;
                resolve();
            });
        });
    }

    /**
     * Ping `llamaServerURL` until we get a success, indicating the HTTP server is running
     */
    async #pollServer(): Promise<void> {
        return new Promise<void>(async (resolve, reject) => {
            process.stdout.write(chalk.dim('waiting for llama-server to load model'));

            let retriesLeft = NUM_RETRIES;

            while (retriesLeft !== 0) {
                if (!this.llamaProcess) {
                    reject(`LlamaManager.#pollServer: llama-server died while polling`);
                    return;
                }

                // Add a '.' to the 'waiting...' output for each time we ping
                process.stdout.write(chalk.dim('.'));

                // Create per-request timeout
                const reqCtrl = new AbortController();
                const reqTimeout = setTimeout(() => reqCtrl.abort(), POLL_TIMEOUT_MS);
                reqTimeout.unref();

                let success = false;
                try {
                    const res = await fetch(this.llamaServerURL, { signal: reqCtrl.signal });
                    success = res.ok;
                } catch {
                    // swallow err - we expect failure while the server is still booting up
                } finally {
                    // Clean up timeout
                    clearTimeout(reqTimeout);

                    if (success) {
                        process.stdout.write(chalk.green('done!\n'));
                        console.log(`llama-server is up, running on port ${chalk.cyan(this.llamaServerPort)}`);
                        resolve();
                        return;
                    }
                }

                // sleep half a second, then retry
                await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS)).then(() => undefined);
                retriesLeft--;
            }
        });
    }
}