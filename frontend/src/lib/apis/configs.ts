import { API_BASE_URL } from '$lib/constants';
import type { ConfigResponse, ConfigUpdateForm } from '@backend/routes/types';

export const getConfig = async (): Promise<ConfigResponse> => {
    const route = '/configs/';
    const res = await fetch(`${API_BASE_URL}${route}`, {
        method: 'GET',
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
        }
    });

    if (!res.ok) {
        const err = await res.json();
        throw err.detail ?? `Request failed: ${route}`;
    }

    return await res.json();
};

export const updateConfig = async (token: string, body: ConfigUpdateForm): Promise<ConfigResponse> => {
    const route = '/configs/';
    const res = await fetch(`${API_BASE_URL}${route}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body)
    });

    if (!res.ok) {
        const err = await res.json();
        throw err.detail ?? `Request failed: ${route}`;
    }

    return await res.json();
};
