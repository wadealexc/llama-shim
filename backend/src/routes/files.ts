import { Router, type Response } from 'express';
import { PDFParse } from 'pdf-parse';
import mammoth from 'mammoth';
import multer from 'multer';

import type { ChatMessageFile } from './types/chats.js';
import type { ErrorResponse, TypedRequest } from './types/common.js';
import { requireAuth } from './middleware.js';

const router = Router();

/* -------------------- FILE VALIDATION CONFIG -------------------- */

// Maximum file size (50MB, aligned with Express body parser limit)
const MB_IN_BYTES = 1024 * 1024;
const MAX_FILE_SIZE = 50 * MB_IN_BYTES;
const MAX_FILE_COUNT = 1;

/* -------------------- FILE CLASSIFICATION -------------------- */

/**
 * Classify a file buffer and return a ChatMessageFile shape.
 * Returns null for unsupported file types.
 */
async function classify(filename: string, mimeType: string, buf: Buffer): Promise<ChatMessageFile | null> {
    const meta = { name: filename, size: buf.length, contentType: mimeType };

    // Images: trust browser mimetype.
    if (mimeType.startsWith('image/')) {
        return {
            kind: 'image',
            dataUrl: `data:${mimeType};base64,${buf.toString('base64')}`,
            ...meta,
        };
    }

    // PDFs: extract text.
    if (mimeType === 'application/pdf') {
        const text = await new PDFParse({ data: buf })
            .getText()
            .then(r => r.text)
            .catch(() => null);

        if (text === null) return null;
        return { kind: 'text', content: text, ...meta, };
    }

    // docx: extract text.
    if (mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document') {
        const text = await mammoth.extractRawText({ buffer: buf })
            .then(r => r.value)
            .catch(() => null);

        if (text === null) return null;
        return { kind: 'text', content: text, ...meta, };
    }

    // Text-like by mimetype OR by content sniff (catches extensionless text,
    // .ts/.md/.yml/etc. that browsers report with weird or missing mimetypes,
    // and unknown programming-language extensions).
    if (mimeType.startsWith('text/') || isLikelyText(buf)) {
        return { kind: 'text', content: buf.toString('utf-8'), ...meta, };
    }

    return null;
}

/**
 * Check if a buffer is likely text content by sampling the first 8KB.
 * Returns true if >95% of bytes are printable ASCII or common whitespace.
 */
function isLikelyText(buf: Buffer): boolean {
    if (buf.length === 0) return false;
    const sample = buf.subarray(0, Math.min(buf.length, 8192));
    let printable = 0;
    for (const b of sample) {
        if (b === 0) return false;
        if ((b >= 32 && b < 127) || b === 9 || b === 10 || b === 13) printable++;
    }
    return printable / sample.length > 0.95;
}

/* -------------------- MULTER MIDDLEWARE -------------------- */

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: MAX_FILE_SIZE,
        files: MAX_FILE_COUNT,
    },
});

/* -------------------- FILE EXTRACTION -------------------- */

/**
 * POST /api/v1/files/extract
 * Access Control: Any verified user
 *
 * Extract and classify a file, returning inline data (base64 for images,
 * extracted text for documents/text files).
 *
 * @body multipart form-data with 'file' field
 * @returns {Types.ChatMessageFile} - inline file data
 */
router.post('/extract', requireAuth, upload.single('file'), async (
    multerReq,
    res: Response<ChatMessageFile | ErrorResponse>
) => {
    const req = multerReq as unknown as TypedRequest<{}, {}, {}>;

    if (!req.file)
        return res.status(400).json({ detail: 'File required' });

    const filename = req.file.originalname;
    const mimeType = req.file.mimetype;
    const buffer = req.file.buffer;

    try {
        const file = await classify(filename, mimeType, buffer);
        
        if (file === null)
            return res.status(400).json({ detail: 'Unsupported file type' });

        return res.json(file);
    } catch (error) {
        console.error('File extraction error:', error);
        return res.status(500).json({ detail: 'File extraction failed' });
    }
});

export default router;
