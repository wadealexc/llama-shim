<script lang="ts">
    import { settings } from '$lib/stores';
    import { onMount } from 'svelte';
    import { setTextScale } from '$lib/utils/text-scale';

    import Minus from '$lib/components/icons/Minus.svelte';
    import Plus from '$lib/components/icons/Plus.svelte';
    import Switch from '$lib/components/common/Switch.svelte';

    export let saveSettings: Function;

    // Interface settings
    let regenerateMenu = true;
    let scrollOnBranchChange = true;
    let ctrlEnterToSend = false;
    let temporaryChatByDefault = false;
    let textScale: number | null = 1;

    const togglectrlEnterToSend = async () => {
        ctrlEnterToSend = !ctrlEnterToSend;
        saveSettings({ ctrlEnterToSend });
    };

    const setTextScaleHandler = (scale: number) => {
        textScale = scale;
        setTextScale(textScale);
        saveSettings({ textScale });
    };

    onMount(async () => {
        regenerateMenu = $settings.regenerateMenu;
        scrollOnBranchChange = $settings.scrollOnBranchChange;
        temporaryChatByDefault = $settings.temporaryChatByDefault;
        ctrlEnterToSend = $settings.ctrlEnterToSend;
        textScale = $settings.textScale;
    });
</script>

<div
    id="tab-interface"
    class="flex flex-col h-full justify-between text-sm"
    role="tabpanel"
>
    <div class="overflow-y-scroll max-h-[28rem] md:max-h-full">
        <h1 class="mb-2 text-sm font-medium">{'Interface'}</h1>

        <!-- Scale -->
        <div class="py-0.5 flex w-full justify-between">
            <label id="ui-scale-label" class="self-center text-xs" for="ui-scale-slider">
                {'Scale'}
            </label>

            <div class="flex items-center gap-2 p-1">
                <button
                    class="text-xs"
                    aria-live="polite"
                    type="button"
                    on:click={() => {
                        if (textScale === null) {
                            textScale = 1;
                        } else {
                            textScale = null;
                            setTextScaleHandler(1);
                        }
                    }}
                >
                    {#if textScale === null}
                        <span>{'Default'}</span>
                    {:else}
                        <span>{textScale}x</span>
                    {/if}
                </button>
            </div>
        </div>

        {#if textScale !== null}
            <div class="flex items-center gap-2 px-1 pb-1">
                <button
                    type="button"
                    class="rounded-lg p-1 transition outline-gray-200 hover:bg-gray-100 dark:outline-gray-700 dark:hover:bg-gray-800"
                    on:click={() => {
                        textScale = Math.max(1, parseFloat((textScale! - 0.1).toFixed(2)));
                        setTextScaleHandler(textScale);
                    }}
                    aria-labelledby="ui-scale-label"
                    aria-label="Decrease UI Scale"
                >
                    <Minus className="h-3.5 w-3.5" />
                </button>

                <div class="flex-1 flex items-center">
                    <input
                        id="ui-scale-slider"
                        class="w-full"
                        type="range"
                        min="1"
                        max="1.5"
                        step={0.01}
                        bind:value={textScale}
                        on:change={() => {
                            setTextScaleHandler(textScale!);
                        }}
                        aria-labelledby="ui-scale-label"
                        aria-valuemin="1"
                        aria-valuemax="1.5"
                        aria-valuenow={textScale}
                        aria-valuetext={`${textScale}x`}
                    />
                </div>

                <button
                    type="button"
                    class="rounded-lg p-1 transition outline-gray-200 hover:bg-gray-100 dark:outline-gray-700 dark:hover:bg-gray-800"
                    on:click={() => {
                        textScale = Math.min(
                            1.5,
                            parseFloat((textScale! + 0.1).toFixed(2))
                        );
                        setTextScaleHandler(textScale);
                    }}
                    aria-labelledby="ui-scale-label"
                    aria-label="Increase UI Scale"
                >
                    <Plus className="h-3.5 w-3.5" />
                </button>
            </div>
        {/if}

        <!-- Temporary chat by default -->
        <div class="py-0.5 flex w-full justify-between">
            <div id="temp-chat-default-label" class="self-center text-xs">
                {'Temporary chat by default'}
            </div>

            <div class="flex items-center gap-2 p-1">
                <Switch
                    ariaLabelledbyId="temp-chat-default-label"
                    tooltip={true}
                    bind:state={temporaryChatByDefault}
                    on:change={() => {
                        saveSettings({ temporaryChatByDefault });
                    }}
                />
            </div>
        </div>

        <!-- Confirm regenerate message -->
        <div class="py-0.5 flex w-full justify-between">
            <div id="regenerate-menu-label" class="self-center text-xs">
                {'Confirm regenerate message'}
            </div>

            <div class="flex items-center gap-2 p-1">
                <Switch
                    ariaLabelledbyId="regenerate-menu-label"
                    tooltip={true}
                    bind:state={regenerateMenu}
                    on:change={() => {
                        saveSettings({ regenerateMenu });
                    }}
                />
            </div>
        </div>

        <!-- Scroll on branch change -->
        <div class="py-0.5 flex w-full justify-between">
            <div id="scroll-on-branch-change-label" class="self-center text-xs">
                {'Scroll on branch change'}
            </div>

            <div class="flex items-center gap-2 p-1">
                <Switch
                    ariaLabelledbyId="scroll-on-branch-change-label"
                    tooltip={true}
                    bind:state={scrollOnBranchChange}
                    on:change={() => {
                        saveSettings({ scrollOnBranchChange });
                    }}
                />
            </div>
        </div>

        <!-- Enter key behavior -->
        <div class="py-0.5 flex w-full justify-between">
            <div
                id="enter-key-behavior-label ctrl-enter-to-send-state"
                class="self-center text-xs"
            >
                {'Enter key behavior'}
            </div>

            <button
                aria-labelledby="enter-key-behavior-label"
                class="p-1 px-3 text-xs flex rounded transition"
                on:click={() => {
                    togglectrlEnterToSend();
                }}
                type="button"
            >
                <span class="ml-2 self-center" id="ctrl-enter-to-send-state"
                    >{ctrlEnterToSend === true ? 'Ctrl+Enter to Send' : 'Enter to Send'}</span
                >
            </button>
        </div>
    </div>
</div>
