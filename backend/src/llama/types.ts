import { Headers } from 'node-fetch';

import * as proto from '../protocol/index.js';
import LlamaStream from './llamaStream.js';

/* -------------------- REQUEST / RESPONSE -------------------- */

export type LlamaRequest = {
    body: proto.CompletionRequest,
    signal: AbortSignal,
    emit?: {
        onQueue: () => void,
        onLoading: () => void,
    },
};

export type LlamaResponse = {
    status: number,
    headers: Headers,
    stream: LlamaStream,
};

/* -------------------- LLAMA INTERFACE -------------------- */

/**
 * Common interface for completion backends. Implemented by `LlamaManager` (which
 * shells out to llama.cpp processes) and `RoutedLlama` (which forwards requests
 * to a remote kitsu backend, used in beta/dev environments to avoid running
 * llama.cpp locally).
 */
export type LlamaStatus = 'idle' | 'queued' | 'active';

export interface Llama {
    completions(req: LlamaRequest): Promise<LlamaResponse>;
    tokenize(content: string, model: string, signal: AbortSignal): Promise<number>;
    wake(modelName: string): LlamaStatus;
    getStatus(model: string): LlamaStatus;
    getModelInfo(name: string): proto.ModelInfo | undefined;
    getAllModelNames(): string[];
    stopServer(): Promise<void>;
    forceStopServer(): void;
}
