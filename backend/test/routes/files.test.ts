import { describe, test, beforeEach } from 'node:test';
import assert from 'node:assert';
import request from 'supertest';
import express, { type Express } from 'express';
import cookieParser from 'cookie-parser';

import { createUserWithToken } from '../helpers.js';
import filesRouter from '../../src/routes/files.js';

/* -------------------- TEST SETUP -------------------- */

// Create Express app with files routes
// Note: Don't use express.json() - it interferes with multer
const app: Express = express();
app.use(cookieParser());
app.use('/api/v1/files', filesRouter);

let token: string;

beforeEach(async () => {
    const result = await createUserWithToken('user');
    token = result.token;
});

describe('POST /api/v1/files/extract', () => {
    test('should extract image and return kind: image with valid dataUrl', async () => {
        // Create a minimal PNG image (1x1 transparent pixel)
        const pngBuffer = Buffer.from([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
            0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, // IEND chunk
            0x42, 0x60, 0x82
        ]);

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', pngBuffer, 'test.png');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.kind, 'image');
        assert.strictEqual(res.body.name, 'test.png');
        assert.strictEqual(res.body.contentType, 'image/png');
        assert.ok(res.body.dataUrl.startsWith('data:image/png;base64,'));
    });

    test('should extract text file and return kind: text with content', async () => {
        const textContent = 'Hello, World!\nThis is a test file.';
        const textBuffer = Buffer.from(textContent);

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', textBuffer, 'test.txt');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.kind, 'text');
        assert.strictEqual(res.body.name, 'test.txt');
        assert.strictEqual(res.body.contentType, 'text/plain');
        assert.strictEqual(res.body.content, textContent);
    });

    test('should extract JSON file and return kind: text', async () => {
        const jsonContent = JSON.stringify({ test: 'data', number: 42 });
        const jsonBuffer = Buffer.from(jsonContent);

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', jsonBuffer, 'test.json');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.kind, 'text');
        assert.ok(res.body.content.includes('test'));
    });

    test('should accept file with no extension but text content via sniff', async () => {
        const textContent = 'Plain text without extension';
        const buffer = Buffer.from(textContent);

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', buffer, 'noextension');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.kind, 'text');
        assert.strictEqual(res.body.content, textContent);
    });

    test('should reject binary file with no recognized type', async () => {
        // Create binary content that won't match text sniffing
        const binaryBuffer = Buffer.from([
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
        ]);

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', binaryBuffer, 'binary.bin');

        assert.strictEqual(res.status, 400);
        assert.strictEqual(res.body.detail, 'Unsupported file type');
    });

    test('should require authentication', async () => {
        const textBuffer = Buffer.from('test');

        const res = await request(app)
            .post('/api/v1/files/extract')
            .attach('file', textBuffer, 'test.txt');

        assert.strictEqual(res.status, 401);
    });

    test('should reject missing file', async () => {
        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`);

        assert.strictEqual(res.status, 400);
        assert.strictEqual(res.body.detail, 'File required');
    });

    test('should reject empty file', async () => {
        const emptyBuffer = Buffer.from('');

        const res = await request(app)
            .post('/api/v1/files/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', emptyBuffer, 'empty.txt');

        // Empty file will fail content sniffing
        assert.strictEqual(res.status, 400);
        assert.strictEqual(res.body.detail, 'Unsupported file type');
    });

    test('should extract PDF file and return kind: text with content', async () => {
        // Minimal valid PDF (1.4 spec)
        const pdfBuffer = Buffer.from(
            '%PDF-1.4\n' +
            '1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n' +
            '2 0 obj<</Type/Pages/MediaBox[0 0 612 792]/Count 1/Kids[3 0 R]>>endobj\n' +
            '3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<</Font<</F1 4 0 R>>>>/Contents 5 0 R>>endobj\n' +
            '4 0 obj<</Type/Font/SubType/Type1/BaseFont/Courier>>endobj\n' +
            '5 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\nendstream\nendobj\n' +
            'xref\n0 6\ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF'
        );

        const res = await request(app)
            .post('/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', pdfBuffer, 'test.pdf');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.kind, 'text');
        assert.strictEqual(res.body.name, 'test.pdf');
        assert.ok(res.body.content.length > 0);
    });

    test('should reject oversized file (>50MB) with 413', async () => {
        // Create a 51MB buffer
        const largeBuffer = Buffer.alloc(51 * 1024 * 1024, 0x42);

        const res = await request(app)
            .post('/extract')
            .set('Authorization', `Bearer ${token}`)
            .attach('file', largeBuffer, 'large.bin');

        assert.strictEqual(res.status, 413);
    });
});
