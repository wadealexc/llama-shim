<script lang="ts">
    import { createEventDispatcher } from 'svelte';

    import { formatFileSize } from '$lib/utils';

    import FileItemModal from './FileItemModal.svelte';
    import Spinner from '$lib/components/common/Spinner.svelte';
    import Tooltip from '$lib/components/common/Tooltip.svelte';
    import XMark from '$lib/components/icons/XMark.svelte';

    const dispatch = createEventDispatcher();

    export let dismissible = false;
    export let loading = false;

    export let item: ChatMessageFile;
    export let small = false;

    import DocumentPage from '$lib/components/icons/DocumentPage.svelte';
    import type { ChatMessageFile } from '@backend/routes/types';

    let showModal = false;
</script>

{#if item.kind === 'text'}
    <FileItemModal bind:show={showModal} bind:item />
{/if}

<div class="relative group w-60">
    <button
        class="p-1.5 w-full flex items-center gap-1 bg-white dark:bg-gray-850 border border-gray-50/30 dark:border-gray-800/30
            {small ? 'rounded-xl p-2' : 'rounded-2xl'} text-left"
        type="button"
        on:click={() => {
            if (item.kind === 'text') {
                showModal = !showModal;
            }
            dispatch('click');
        }}
    >
        {#if !small}
            <div
                class="size-10 shrink-0 flex justify-center items-center bg-black/20 dark:bg-white/10 text-white rounded-xl"
            >
                {#if !loading}
                    <DocumentPage />
                {:else}
                    <Spinner />
                {/if}
            </div>
        {:else}
            <div class="pl-1.5">
                {#if !loading}
                    <Tooltip content="Document" placement="top">
                        <DocumentPage />
                    </Tooltip>
                {:else}
                    <Spinner />
                {/if}
            </div>
        {/if}

        {#if !small}
            <div class="flex flex-col justify-center -space-y-0.5 px-2.5 w-full">
                <div class=" dark:text-gray-100 text-sm font-medium line-clamp-1 mb-1">
                    {item.name}
                </div>

                <div class=" flex justify-between text-xs line-clamp-1 text-gray-500">
                    {'Document'}
                    {#if item.size}
                        <span class="capitalize">{formatFileSize(item.size)}</span>
                    {/if}
                </div>
            </div>
        {:else}
            <Tooltip content={item.name} className="flex flex-col w-full" placement="top-start">
                <div class="flex flex-col justify-center -space-y-0.5 px-1 w-full">
                    <div class=" dark:text-gray-100 text-sm flex justify-between items-center">
                        <div class="font-medium line-clamp-1 flex-1 pr-1">
                            {item.name}
                        </div>
                        {#if item.size}
                            <div class="text-gray-500 text-xs capitalize shrink-0">
                                {formatFileSize(item.size)}
                            </div>
                        {/if}
                    </div>
                </div>
            </Tooltip>
        {/if}
    </button>
    {#if dismissible}
        <div class=" absolute -top-1 -right-1">
            <button
                aria-label="Remove File"
                class=" bg-white text-black border border-gray-50 rounded-full outline-hidden focus:outline-hidden group-hover:visible invisible transition"
                type="button"
                on:click|stopPropagation={() => {
                    dispatch('dismiss');
                }}
            >
                <XMark className="size-4" />
            </button>
        </div>
    {/if}
</div>
