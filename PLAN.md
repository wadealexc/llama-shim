# Fix Settings Persistence Bug + Change Settings Update to Merge

## Context

User settings (e.g. `webSearch: true`) keep getting silently reset to defaults. Two root causes:

1. **Missing `loadUserSettings()` after sign-in**: When a token is expired, the root layout redirects to `/auth` without loading settings. After sign-in, `goto('/')` is client-side navigation — `loadUserSettings()` never re-runs. The settings store retains `DEFAULT_SETTINGS` (`webSearch: false`).

2. **`getUserSettings` API failure fallback**: If the API call fails, fallback reads `localStorage.getItem('settings')` which is always empty (settings are never written to localStorage), producing defaults.

Both are amplified by **ModelSelector's auto-save** (`ModelSelector.svelte:26-30`), which persists the *entire* settings object to the backend whenever the selected model changes. If `$settings` has defaults at that moment, the backend data is overwritten.

The fix has two parts:
- **Part A**: Fix frontend initialization gaps so `$settings` is always correct before auto-save can fire
- **Part B**: Change the existing `settings/update` endpoint from full-replace to shallow-merge, and update frontend callers to only send the fields they actually changed

---

## Part A: Fix Frontend Initialization

### A1. Extract `loadUserSettings` into a shared utility

**Create** `frontend/src/lib/utils/settings.ts`

Extract the settings-loading logic from `+layout.svelte:66-84` into a reusable function. Key change from current behavior: on API failure, do NOT fall back to localStorage (which is always empty) — just leave the store as-is. This prevents overwriting good settings with defaults on a transient network error.

```typescript
import { getUserSettings } from '$lib/apis/users';
import { settings, applySettingsDefaults } from '$lib/stores';
import { get } from 'svelte/store';
import { setTextScale } from '$lib/utils/text-scale';

export async function loadUserSettings(): Promise<void> {
    const userSettings = await getUserSettings(localStorage.token).catch((error) => {
        console.error('Failed to load user settings:', error);
        return null;
    });

    if (!userSettings) return; // leave store as-is on failure

    settings.set(applySettingsDefaults(userSettings.ui));
    setTextScale(get(settings).textScale);
}
```

### A2. Update callers to use shared utility

**Modify** `frontend/src/routes/+layout.svelte` (lines 66-84): Replace the local `loadUserSettings` function with an import from `$lib/utils/settings`. Remove the `getUserSettings` import (no longer used directly). Remove the `applySettingsDefaults` import (no longer used directly). Remove the `setTextScale` import (no longer used directly).

**Modify** `frontend/src/routes/s/[id]/+page.svelte` (lines 54-71): Replace the duplicated inline settings-loading logic with `await loadUserSettings()` from `$lib/utils/settings`. Remove the `getUserSettings` import and `applySettingsDefaults` import (no longer used directly).

**Modify** `frontend/src/routes/auth/+page.svelte` (lines 22-40): In `setSessionUser`, after setting token and user store but *before* `goto()`, call `await loadUserSettings()`. This ensures settings are populated before navigation causes ModelSelector to render.

### A3. Fix ModelSelector reactive dependency on `$settings`

**Modify** `frontend/src/lib/components/chat/ModelSelector.svelte` (lines 25-30)

The current auto-save reads `$settings` in the reactive block, making Svelte track it as a dependency — the block re-runs on *any* settings change. Use `settings.update()` to avoid the reactive read, and only send `{ ui: { model } }` (which the merge endpoint will handle correctly):

```typescript
// Before:
$: if (initialized && value !== '') {
    const newSettings = { ...$settings, model: value };
    settings.set(newSettings);
    updateUserSettings(localStorage.token, { ui: newSettings });
}

// After:
$: if (initialized && value !== '') {
    settings.update(s => ({ ...s, model: value }));
    updateUserSettings(localStorage.token, { ui: { model: value } });
}
```

---

## Part B: Change Settings Update to Merge + Migrate Callers

### B1. Change `updateUserSettings` DB operation to merge

**Modify** `backend/src/db/operations/users.ts` — replace `updateUserSettings` (lines 125-142):

Read existing settings, shallow-merge at the `ui` level (`{ ...existing.ui, ...input.ui }`), write back. Uses `getUserById` (same file) to read current state.

```typescript
export async function updateUserSettings(
    id: string,
    settings: UserSettings,
    txOrDb: DbOrTx = db
): Promise<UserSettings> {
    const existingUser = await getUserById(id, txOrDb);
    if (!existingUser) throw new RecordNotFoundError(TABLE, id);

    const existing = existingUser.settings ?? { ui: {} };

    const merged: UserSettings = {
        ...existing,
        ...settings,
        ui: {
            ...(existing.ui ?? {}),
            ...(settings.ui ?? {}),
        },
    };

    const [user] = await txOrDb
        .update(users)
        .set({ settings: merged, updatedAt: currentUnixTimestamp() })
        .where(eq(users.id, id))
        .returning();

    if (!user) throw new RecordNotFoundError(TABLE, id);
    if (!user.settings) throw new DatabaseError('Settings not found after update');
    return user.settings;
}
```

No route changes needed — `POST /api/v1/users/user/settings/update` stays the same, just its behavior changes from replace to merge.

### B2. Migrate frontend callers to send only changed fields

These callers currently send the entire settings object but only change one or two fields. With the merge endpoint, they can send just what they changed:

| File | What it changes | Before | After |
|------|----------------|--------|-------|
| `ModelSelector.svelte:26-30` | `model` | `{ ui: { ...allSettings, model } }` | `{ ui: { model } }` |
| `ModelSelector.svelte:32-43` | `pinnedModels` | `{ ui: $settings }` | `{ ui: { pinnedModels } }` |
| `PinnedModelList.svelte:47-53` | `pinnedModels` | `{ ui: $settings }` | `{ ui: { pinnedModels } }` |
| `Models.svelte:73-84` | `pinnedModels` | `{ ui: $settings }` | `{ ui: { pinnedModels } }` |
| **`SettingsModal.svelte:29-34`** | **everything** | `{ ui: $settings }` | **No change needed** — sending all fields still works with merge |

For all migrated callers, also switch from `settings.set({ ...$settings, field })` to `settings.update(s => ({ ...s, field }))` to avoid reactive reads of `$settings`.

### B3. Update backend tests

**Modify** `backend/test/db/operations/users.test.ts` — update `describe('updateUserSettings')` (line 257):

Update existing test and add new ones for merge behavior:
- Merges into empty/null settings
- Preserves existing `ui` fields when merging new ones
- Overwrites specified `ui` fields, leaves others intact
- Throws for non-existent user

**Modify** `backend/test/routes/users.test.ts` — update `describe('POST .../settings/update')` (line 272):

Update the "should replace existing settings" test (line 295) to verify merge behavior instead:
- Old: verifies full replacement
- New: verifies partial merge preserves existing fields

Add new tests:
- Merges partial `ui` fields, preserves others (200) — verify both response and DB
- Merges `pinnedModels` without affecting other fields (200)

---

## Implementation Order

1. **B1** — change `updateUserSettings` DB operation to merge
2. **B3** — update backend tests for merge behavior
3. **A1** — create shared `loadUserSettings` utility
4. **A2** — update `+layout.svelte`, `auth/+page.svelte`, `s/[id]/+page.svelte`
5. **A3 + B2** — fix ModelSelector reactivity + migrate all callers to send partial fields

## Files Changed

| File | Action |
|------|--------|
| `frontend/src/lib/utils/settings.ts` | **Create** — shared `loadUserSettings` helper |
| `backend/src/db/operations/users.ts` | Modify — change `updateUserSettings` to merge |
| `backend/test/db/operations/users.test.ts` | Modify — update + add tests for merge behavior |
| `backend/test/routes/users.test.ts` | Modify — update + add tests for merge behavior |
| `frontend/src/routes/+layout.svelte` | Modify — import shared `loadUserSettings` |
| `frontend/src/routes/auth/+page.svelte` | Modify — call `loadUserSettings` after sign-in |
| `frontend/src/routes/s/[id]/+page.svelte` | Modify — use shared `loadUserSettings` |
| `frontend/src/lib/components/chat/ModelSelector.svelte` | Modify — fix reactivity + send partial |
| `frontend/src/lib/components/sidebar/PinnedModelList.svelte` | Modify — send partial |
| `frontend/src/lib/components/models/Models.svelte` | Modify — send partial |

## Verification

1. `npm run check` — backend types pass
2. `npm run check:frontend` — frontend types pass
3. `npm run test` — all backend tests pass (updated + new merge tests)
4. Manual: enable webSearch → hard reload → verify it persists
5. Manual: enable webSearch → sign out → sign in → verify it persists
