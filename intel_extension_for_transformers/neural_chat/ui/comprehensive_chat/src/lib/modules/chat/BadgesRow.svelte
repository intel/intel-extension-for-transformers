<script lang="ts">
	import ArrowIcon from "$lib/assets/chat/svelte/ArrowRight.svelte";
	import { Badge } from "flowbite-svelte";
	import { COLORS } from "$lib/shared/constant/Data";
	import { createEventDispatcher } from "svelte";
	import { imageList } from "$lib/shared/stores/common/Store";

    export let badges: string[]

    const dispatch = createEventDispatcher()
    const iconClass = 'mx-1 flex h-5 w-5 items-center justify-center rounded-full border bg-gray-200 show-arrow'

    let container: HTMLElement
    let curIdx = 0
    function leftMove() {
        if (curIdx === badges.length - 1) return
        curIdx++;
        move()
    }

    function rightMove() {
        if (curIdx === 0) return
        curIdx--;
        move()
    }

    function move() {
        const nodes = [...container.children] as HTMLElement[]
        container.scroll({
            behavior: "smooth",
            left: nodes[curIdx].offsetLeft
        })
    }
</script>

{#if $imageList.length > 0}
<div class="flex items-center">
    <button
        class={`rotate-180 ${iconClass}`}
        class:hidden={curIdx === 0}
        on:click={rightMove}
    >
        <ArrowIcon />
    </button>
    <div
        class="relative scrollbar-none scrollbar-hide flex w-screen overflow-x-auto pb-1 gap-2"
        bind:this={container}
    >
        {#each badges as badge, idx (badge)}
            <button on:click={() => dispatch('clickBadge', badge)}>
                <Badge
                    rounded
                    color="blue"
                    class="mx-2 inline-block w-full whitespace-nowrap"
                >
                    {badge}
                </Badge>
            </button>
        {/each}
    </div>
    <button
        class={iconClass}
        class:hidden={curIdx === badges.length - 1}
        on:click={leftMove}
    >
        <ArrowIcon />
    </button>
</div>
{/if}

<style>
	.scrollbar-hide::-webkit-scrollbar {
		display: none;
	}

	@media only screen and (min-width: 1000px) {
		.show-arrow {
			display: none;
		}
	}
</style>