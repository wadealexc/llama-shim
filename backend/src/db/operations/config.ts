import { eq } from 'drizzle-orm';

import { db, type DbOrTx } from '../client.js';
import { config } from '../schema.js';
import { currentUnixTimestamp } from '../utils.js';
import type { UserRole } from '../../routes/types/index.js';
import type { StringValue } from 'ms';
import { RecordNotFoundError } from '../errors.js';

const TABLE = 'config';

/* -------------------- TYPES -------------------- */

export type Config = typeof config.$inferSelect;

export type UpdateConfig = Partial<Pick<Config, 'enableSignup' | 'defaultUserRole' | 'jwtExpiresIn'>>;

/* -------------------- READ -------------------- */

/**
 * Retrieve the single config row.
 * 
 * @param txOrDb
 * 
 * @returns the config row
 * @throws {RecordNotFoundError} if config row does not exist
 */
export async function getConfig(
    txOrDb: DbOrTx = db
): Promise<Config> {
    const [row] = await txOrDb
        .select()
        .from(config)
        .limit(1);

    if (!row) {
        throw new RecordNotFoundError(TABLE, '1');
    }

    return row;
}

/* -------------------- UPDATE -------------------- */

/**
 * Updates the config row.
 * 
 * Automatically updates the `updatedAt` timestamp.
 * 
 * @param {UpdateConfig} updates - Fields to update
 * @param txOrDb
 * 
 * @returns the updated config row
 * @throws {RecordNotFoundError} if config row does not exist
 */
export async function updateConfig(
    updates: UpdateConfig,
    txOrDb: DbOrTx = db
): Promise<Config> {
    const now = currentUnixTimestamp();

    const [updated] = await txOrDb
        .update(config)
        .set({
            ...updates,
            updatedAt: now,
        })
        .where(eq(config.id, 1))
        .returning();

    if (!updated) {
        throw new RecordNotFoundError(TABLE, '1');
    }

    return updated;
}
