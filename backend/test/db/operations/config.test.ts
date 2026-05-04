import { describe, test, before, beforeEach } from 'node:test';
import assert from 'node:assert';

import { createTestDatabase, type TestDatabase } from '../../helpers.js';
import { Configs, schema } from '../../../src/db/index.js';


/* -------------------- CRUD OPERATIONS TESTS -------------------- */

describe('getConfig', () => {
    let db: TestDatabase;

    beforeEach(async () => {
        db = await createTestDatabase();
    });

    test('returns config seeded by migration', async () => {
        const config = await Configs.getConfig(db);

        assert.strictEqual(config.id, 1);
        assert.strictEqual(config.name, 'kitsu');
        assert.strictEqual(config.enableSignup, true);
        assert.strictEqual(config.defaultUserRole, 'user');
        assert.strictEqual(config.jwtExpiresIn, '7d');
        assert.ok(config.updatedAt > 0);
    });

    test('returns correct field types', async () => {
        const config = await Configs.getConfig(db);

        assert.strictEqual(typeof config.id, 'number');
        assert.strictEqual(typeof config.name, 'string');
        assert.strictEqual(typeof config.enableSignup, 'boolean');
        assert.strictEqual(typeof config.defaultUserRole, 'string');
        assert.strictEqual(typeof config.jwtExpiresIn, 'string');
        assert.strictEqual(typeof config.updatedAt, 'number');
    });

    test('returns updated config after update', async () => {
        // Update config
        await Configs.updateConfig({ enableSignup: false }, db);

        const config = await Configs.getConfig(db);

        assert.strictEqual(config.enableSignup, false);
        assert.strictEqual(config.defaultUserRole, 'user'); // unchanged
        assert.strictEqual(config.jwtExpiresIn, '7d'); // unchanged
    });

    test('throws error if config row does not exist', async () => {
        // Delete the config row to simulate missing row
        await db.delete(schema.config);

        await assert.rejects(
            async () => await Configs.getConfig(db),
            { message: 'config record with id \'1\' not found' }
        );
    });
});

describe('updateConfig', () => {
    let db: TestDatabase;

    beforeEach(async () => {
        db = await createTestDatabase();
    });

    test('updates enableSignup', async () => {
        const updated = await Configs.updateConfig({ enableSignup: false }, db);

        assert.strictEqual(updated.enableSignup, false);
        assert.strictEqual(updated.defaultUserRole, 'user'); // unchanged
    });

    test('updates defaultUserRole', async () => {
        const updated = await Configs.updateConfig({ defaultUserRole: 'admin' }, db);

        assert.strictEqual(updated.defaultUserRole, 'admin');
        assert.strictEqual(updated.enableSignup, true); // unchanged
    });

    test('updates jwtExpiresIn', async () => {
        const updated = await Configs.updateConfig({ jwtExpiresIn: '30d' }, db);

        assert.strictEqual(updated.jwtExpiresIn, '30d');
        assert.strictEqual(updated.enableSignup, true); // unchanged
    });

    test('partial update leaves other fields unchanged', async () => {
        // First update enableSignup
        await Configs.updateConfig({ enableSignup: false }, db);

        // Then only update jwtExpiresIn
        const updated = await Configs.updateConfig({ jwtExpiresIn: '14d' }, db);

        assert.strictEqual(updated.enableSignup, false); // preserved
        assert.strictEqual(updated.jwtExpiresIn, '14d'); // updated
        assert.strictEqual(updated.defaultUserRole, 'user'); // unchanged
    });

    test('updates updatedAt timestamp', async () => {
        const config1 = await Configs.getConfig(db);
        const updatedAt1 = config1.updatedAt;

        // Small delay to ensure timestamp changes
        await new Promise(resolve => setTimeout(resolve, 10));

        const updated = await Configs.updateConfig({ enableSignup: false }, db);

        assert.ok(updated.updatedAt >= updatedAt1);
    });

    test('returns updated config row', async () => {
        const updated = await Configs.updateConfig({
            enableSignup: false,
            defaultUserRole: 'pending',
            jwtExpiresIn: '2w',
        }, db);

        assert.strictEqual(updated.id, 1);
        assert.strictEqual(updated.name, 'kitsu');
        assert.strictEqual(updated.enableSignup, false);
        assert.strictEqual(updated.defaultUserRole, 'pending');
        assert.strictEqual(updated.jwtExpiresIn, '2w');
        assert.ok(updated.updatedAt > 0);
    });

    test('throws error if config row does not exist', async () => {
        const freshDb = await createTestDatabase();
        
        // Delete the config row to simulate missing row
        await freshDb.delete(schema.config);

        await assert.rejects(
            async () => await Configs.updateConfig({ enableSignup: false }, freshDb),
            { message: 'config record with id \'1\' not found' }
        );
    });
});
