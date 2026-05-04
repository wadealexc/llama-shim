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
