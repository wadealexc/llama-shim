/**
 * Configuration Routes
 *
 * Handles system-wide configuration management.
 */

import { Router, type Request, type Response } from 'express';
import * as Types from './types/index.js';
import { requireAdmin } from './middleware.js';
import { db, Configs } from '../db/index.js';

const router = Router();

/* -------------------- HELPERS -------------------- */

function toConfigResponse(config: Configs.Config): Types.ConfigResponse {
    return {
        name: config.name,
        enableSignup: config.enableSignup,
        defaultUserRole: config.defaultUserRole,
        jwtExpiresIn: config.jwtExpiresIn,
    };
}

/* -------------------- PUBLIC ENDPOINTS -------------------- */

/**
 * GET /api/v1/configs/
 * Access Control: Public (no authentication required)
 *
 * Returns the current system configuration.
 *
 * @returns {Types.ConfigResponse} - flat config object with name, enableSignup, defaultUserRole, jwtExpiresIn
 */
router.get('/', async (
    _req: Request,
    res: Response<Types.ConfigResponse | Types.ErrorResponse>
) => {
    const config = await Configs.getConfig(db);
    return res.json(toConfigResponse(config));
});

/* -------------------- ADMIN ENDPOINTS -------------------- */

/**
 * POST /api/v1/configs/
 * Access Control: Requires HTTPBearer authentication and admin role
 *
 * Update the system configuration.
 *
 * @param {Types.ConfigUpdateForm} - partial config object with fields to update
 * @returns {Types.ConfigResponse} - full config object after update
 */
router.post('/', requireAdmin, async (
    req: Types.TypedRequest<{}, Types.ConfigUpdateForm>,
    res: Response<Types.ConfigResponse | Types.ErrorResponse>
) => {
    const body = Types.ConfigUpdateFormSchema.safeParse(req.body);
    if (!body.success) {
        return res.status(400).json({
            detail: 'Invalid request body',
            errors: body.error.issues
        });
    }

    const updated = await Configs.updateConfig(body.data, db);
    return res.json(toConfigResponse(updated));
});

/* -------------------- EXPORT -------------------- */

export default router;
