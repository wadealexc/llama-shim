import { describe, test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert';
import request from 'supertest';
import express, { type Express } from 'express';
import cookieParser from 'cookie-parser';

import { assertInMemoryDatabase } from '../helpers.js';
import { db, schema, Configs } from '../../src/db/index.js';
import { eq } from 'drizzle-orm';
import { migrate } from 'drizzle-orm/libsql/migrator';
import authRouter from '../../src/routes/auths.js';
import configsRouter from '../../src/routes/configs.js';

/* -------------------- TEST SETUP -------------------- */

// Ensure tests use in-memory database
assertInMemoryDatabase();

// Apply migrations to the in-memory database (async with libSQL)
await migrate(db, { migrationsFolder: './drizzle' });

// Helper function to clear database tables
async function clearDatabase() {
    await db.delete(schema.auths);
    await db.delete(schema.users);
    // Reset config to defaults rather than deleting (signup needs the config row)
    await db.update(schema.config).set({
        enableSignup: true,
        defaultUserRole: 'user',
        jwtExpiresIn: '7d',
    }).where(eq(schema.config.id, 1));
}

// Create Express app with auth and configs routes
const app: Express = express();
app.use(express.json());
app.use(cookieParser());
app.use('/api/v1/auths', authRouter);
app.use('/api/v1/configs', configsRouter);

/* -------------------- GET /api/v1/configs TESTS -------------------- */

describe('GET /api/v1/configs', () => {
    afterEach(async () => {
        await clearDatabase();
    });

    test('returns config with correct flat shape', async () => {
        const res = await request(app)
            .get('/api/v1/configs');

        assert.strictEqual(res.status, 200);
        assert.ok(res.body);
        assert.strictEqual(typeof res.body.name, 'string');
        assert.strictEqual(typeof res.body.enableSignup, 'boolean');
        assert.strictEqual(typeof res.body.defaultUserRole, 'string');
        assert.strictEqual(typeof res.body.jwtExpiresIn, 'string');
    });

    test('returns defaults on fresh DB', async () => {
        const res = await request(app)
            .get('/api/v1/configs');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.name, 'kitsu');
        assert.strictEqual(res.body.enableSignup, true);
        assert.strictEqual(res.body.defaultUserRole, 'user');
        assert.strictEqual(res.body.jwtExpiresIn, '7d');
    });

    test('works without authentication (public)', async () => {
        const res = await request(app)
            .get('/api/v1/configs');

        assert.strictEqual(res.status, 200);
        assert.ok(res.body);
    });

    test('returns updated values after POST', async () => {
        // Create admin user for POST
        const adminRes = await request(app)
            .post('/api/v1/auths/signup')
            .send({
                username: 'admin',
                password: 'password123',
            });
        const adminToken = adminRes.body.token;

        // Update config
        await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                enableSignup: false,
                defaultUserRole: 'pending',
                jwtExpiresIn: '14d',
            });

        // GET should return updated values
        const res = await request(app)
            .get('/api/v1/configs');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.enableSignup, false);
        assert.strictEqual(res.body.defaultUserRole, 'pending');
        assert.strictEqual(res.body.jwtExpiresIn, '14d');
    });
});

/* -------------------- POST /api/v1/configs TESTS -------------------- */

describe('POST /api/v1/configs', () => {
    let adminToken: string;
    let userToken: string;

    afterEach(async () => {
        await clearDatabase();
    });

    beforeEach(async () => {
        // Create admin user (first user)
        const adminRes = await request(app)
            .post('/api/v1/auths/signup')
            .send({
                username: 'admin',
                password: 'password123',
            });
        adminToken = adminRes.body.token;

        // Create regular user
        const userRes = await request(app)
            .post('/api/v1/auths/signup')
            .send({
                username: 'user',
                password: 'password123',
            });
        userToken = userRes.body.token;
    });

    test('updates config fields (admin auth)', async () => {
        const res = await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                enableSignup: false,
                defaultUserRole: 'pending',
                jwtExpiresIn: '30d',
            });

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.enableSignup, false);
        assert.strictEqual(res.body.defaultUserRole, 'pending');
        assert.strictEqual(res.body.jwtExpiresIn, '30d');
        assert.strictEqual(res.body.name, 'kitsu'); // unchanged
    });

    test('partial update works (only send subset of fields)', async () => {
        // Update only enableSignup
        const res = await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                enableSignup: false,
            });

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.enableSignup, false);
        assert.strictEqual(res.body.defaultUserRole, 'user'); // unchanged default
        assert.strictEqual(res.body.jwtExpiresIn, '7d'); // unchanged default
    });

    test('GET returns updated values after POST', async () => {
        // Update config
        await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                enableSignup: false,
            });

        // GET should return updated value
        const res = await request(app)
            .get('/api/v1/configs');

        assert.strictEqual(res.status, 200);
        assert.strictEqual(res.body.enableSignup, false);
    });

    test('accepts valid JWT expiration formats', async () => {
        const validFormats = ['7d', '4w', '24h', '30m', '-1'];

        for (const format of validFormats) {
            const res = await request(app)
                .post('/api/v1/configs')
                .set('Authorization', `Bearer ${adminToken}`)
                .send({
                    jwtExpiresIn: format,
                });

            assert.strictEqual(res.status, 200, `Failed for format: ${format}`);
            assert.strictEqual(res.body.jwtExpiresIn, format);
        }
    });

    test('rejects invalid JWT expiration format', async () => {
        const res = await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                jwtExpiresIn: 'invalid-format',
            });

        assert.strictEqual(res.status, 400);
        assert.strictEqual(res.body.detail, 'Invalid request body');
    });

    test('rejects invalid user role', async () => {
        const res = await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${adminToken}`)
            .send({
                defaultUserRole: 'superuser', // Invalid role
            });

        assert.strictEqual(res.status, 400);
        assert.strictEqual(res.body.detail, 'Invalid request body');
    });

    test('rejects non-admin user', async () => {
        const res = await request(app)
            .post('/api/v1/configs')
            .set('Authorization', `Bearer ${userToken}`)
            .send({
                enableSignup: false,
            });

        assert.strictEqual(res.status, 403);
        assert.strictEqual(res.body.detail, 'Admin access required');
    });

    test('rejects unauthenticated request', async () => {
        const res = await request(app)
            .post('/api/v1/configs')
            .send({
                enableSignup: false,
            });

        assert.strictEqual(res.status, 401);
        assert.strictEqual(res.body.detail, 'Not authenticated');
    });
});
