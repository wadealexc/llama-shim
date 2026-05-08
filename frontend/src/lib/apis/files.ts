import { API_BASE_URL } from '$lib/constants';
import type { ChatMessageFile } from '@backend/routes/types';

/**
 * Extract and classify a file, returning inline data.
 * For images: returns base64 data URL inline.
 * For text/PDF/docx: returns extracted text content.
 */
export const extractFile = async (token: string, file: File): Promise<ChatMessageFile> => {
    const route = '/files/extract';
    const data = new FormData();
    data.append('file', file);

    const res = await fetch(`${API_BASE_URL}${route}`, {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            Authorization: `Bearer ${token}`,
        },
        body: data
    });

    if (!res.ok) {
        const err = await res.json();
        throw err.detail ?? `Request failed: ${route}`;
    }

    return await res.json();
};
