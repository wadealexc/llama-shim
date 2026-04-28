<script lang="ts">
    // No state needed - pure visual component
</script>

<div class="flex gap-1.5">
    <div class="dot w-1 h-1 bg-gray-600 dark:bg-gray-300 rounded-full"></div>
    <div class="dot w-1 h-1 bg-gray-600 dark:bg-gray-300 rounded-full delay-100"></div>
    <div class="dot w-1 h-1 bg-gray-600 dark:bg-gray-300 rounded-full delay-200"></div>
</div>

<style>
    .dot {
        animation: typing 1.4s infinite;
        opacity: 0;
    }

    .delay-100 {
        animation-delay: 100ms;
    }

    .delay-200 {
        animation-delay: 200ms;
    }

    @keyframes typing {
        0%, 100% {
            opacity: 0;
        }
        25% {
            opacity: 1;
        }
        50%, 75% {
            opacity: 1;
        }
    }
</style>
