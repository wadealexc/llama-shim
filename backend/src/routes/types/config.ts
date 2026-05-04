import { z } from 'zod';
import parse, { type StringValue } from 'ms';

import { UserRoleSchema } from './common.js';

// JWT expiration string validation
const MsStringValueSchema = z.custom<StringValue>((v) => {
    if (v === '-1') return true;

    try {
        const n = parse(v as StringValue);
        return typeof n === "number" && Number.isFinite(n);
    } catch {
        return false;
    }
}, {
    message: 'Must be a valid ms time string (e.g. "1d", "2h", "30m", "2 days", "1 mo")',
});

// Config response (flat shape)
export const ConfigResponseSchema = z.object({
    name: z.string(),
    enableSignup: z.boolean(),
    defaultUserRole: UserRoleSchema,
    jwtExpiresIn: MsStringValueSchema,
});
export type ConfigResponse = z.infer<typeof ConfigResponseSchema>;

// Config update form (partial - all fields optional)
export const ConfigUpdateFormSchema = z.object({
    enableSignup: z.boolean().optional(),
    defaultUserRole: UserRoleSchema.optional(),
    jwtExpiresIn: MsStringValueSchema.optional(),
});
export type ConfigUpdateForm = z.infer<typeof ConfigUpdateFormSchema>;
