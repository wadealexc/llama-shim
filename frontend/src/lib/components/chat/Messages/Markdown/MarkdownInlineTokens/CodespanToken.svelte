<script lang="ts">
    import { copyToClipboard, unescapeHtml } from '$lib/utils';
    import { toast } from 'svelte-sonner';
    import type { Tokens } from 'marked';

    export let token: Tokens.Codespan;
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
<code
    class="codespan cursor-pointer"
    on:click={() => {
        copyToClipboard(unescapeHtml(token.text));
        toast.success('Copied to clipboard');
    }}
>
    {unescapeHtml(token.text)}
</code>
