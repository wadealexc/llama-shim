import type { FolderNameIdResponse, FolderData } from '@backend/routes/types';

export type SidebarFolder = FolderNameIdResponse & {
    childrenIds?: string[];
    new?: boolean;
    data?: FolderData;
};
