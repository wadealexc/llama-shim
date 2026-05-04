import { z } from 'zod';

import { UserIdSchema, UserRoleSchema } from './common.js';

// Signin
export const SigninFormSchema = z.object({
    username: z.string(),
    password: z.string(),
});
export type SigninForm = z.infer<typeof SigninFormSchema>;

// Signup
export const SignupFormSchema = z.object({
    username: z.string(),
    password: z.string(),
});
export type SignupForm = z.infer<typeof SignupFormSchema>;

// Signout
export const SignoutResponseSchema = z.object({
    status: z.boolean(),
    redirectUrl: z.string().nullable().optional(),
});
export type SignoutResponse = z.infer<typeof SignoutResponseSchema>;

// Session responses
export const SessionUserResponseSchema = z.object({
    id: UserIdSchema,
    username: z.string(),
    role: UserRoleSchema,
    token: z.string(),
    tokenType: z.string(),
    expiresAt: z.number().nullable().optional(),
});
export type SessionUserResponse = z.infer<typeof SessionUserResponseSchema>;

// Profile update
export const UpdateProfileFormSchema = z.object({
    username: z.string(),
});
export type UpdateProfileForm = z.infer<typeof UpdateProfileFormSchema>;

export const UpdateProfileResponseSchema = z.object({
    id: UserIdSchema,
    username: z.string(),
    role: UserRoleSchema,
});
export type UpdateProfileResponse = z.infer<typeof UpdateProfileResponseSchema>;

// Password update
export const UpdatePasswordFormSchema = z.object({
    password: z.string(),
    newPassword: z.string(),
});
export type UpdatePasswordForm = z.infer<typeof UpdatePasswordFormSchema>;
