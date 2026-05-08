<script lang="ts">
    import { formatFileSize, getLineCount } from '$lib/utils';

    import Modal from '$lib/components/common/Modal.svelte';
    import XMark from '$lib/components/icons/XMark.svelte';

    import type { ChatMessageFile } from '@backend/routes/types';

    export let item: Extract<ChatMessageFile, { kind: 'text' }>;
    export let show = false;
</script>

<Modal bind:show size="lg">
    <div class="font-primary px-4.5 py-3.5 w-full flex flex-col justify-center dark:text-gray-400">
        <div class="pb-2">
            <div class="flex items-start justify-between">
                <div>
                    <div class="font-medium text-lg dark:text-gray-100">
                        {item.name}
                    </div>
                </div>

                <div>
                    <button
                        on:click={() => {
                            show = false;
                        }}
                    >
                        <XMark />
                    </button>
                </div>
            </div>

            <div>
                <div class="flex flex-col items-center md:flex-row gap-1 justify-between w-full">
                    <div class="flex flex-wrap text-xs gap-1 text-gray-500">
                        {#if item.size}
                            <div class="capitalize shrink-0">{formatFileSize(item.size)}</div>
                        {/if}

                        <div class="capitalize shrink-0">
                            {`${getLineCount(item.content ?? '')} lines`}
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="max-h-[75vh] overflow-auto">
            <div class="max-h-96 overflow-scroll scrollbar-hidden text-xs whitespace-pre-wrap">
                {item.content.trim() || 'No content'}
            </div>
        </div>
    </div>
</Modal>
