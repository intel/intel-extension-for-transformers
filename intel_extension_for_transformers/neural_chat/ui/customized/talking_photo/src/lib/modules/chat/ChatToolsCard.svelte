<script lang="ts">
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import { toolList } from "$lib/shared/constant/Data";
	import { createEventDispatcher } from "svelte";
	import { Button, Modal } from "flowbite-svelte";

	export let extraClass = "";
	let id: string;
	let open = false;
	const dispatch = createEventDispatcher();

	const setPlacement = (title: string) => {
		id = `${title}`;
		open = !open;
	};
</script>

<div class="{`${extraClass}`} w-full">
	<Scrollbar
		className="h-44 sm:grow p-4"
		classLayout="grid grid-cols-8 max-sm:grid-cols-4 gap-3"
	>
		{#each toolList as tool, idx}
			<div
				class="flex max-w-sm flex-col items-center bg-white text-center text-gray-500 max-sm:py-2"
			>
				<button
					class="flex w-14 justify-center rounded-lg border border-gray-200 py-3 shadow-md max-sm:w-14 h-14 items-center"
					on:click={() => dispatch("showTool", tool.title)}
				>
					<div><svelte:component this={tool.icon} /></div>
				</button>
				<span class="pt-1 text-xs">
					{tool.title}
				</span>
			</div>
		{/each}
	</Scrollbar>
</div>

<!-- <Modal {id} title={id} bind:open autoclose  class="fixed bottom-0 left-0 right-0 z-50  w-full p-4 overflow-y-auto md:inset-0 h-[calc(100%-20rem)] max-h-full">
	{#if id === "Hint"}
		<ul class="flex flex-col gap-3 p-3">
			<li>
				<button
					type="button"
					class="w-full text-left text-sm font-semibold tracking-wide focus:outline-none"
					>Recomended</button
				>
			</li>
			<li>
				<button
					type="button"
					class="w-full text-left text-sm font-semibold tracking-wide focus:outline-none"
					>What's New</button
				>
			</li>
			<li>
				<button
					type="button"
					class="w-full text-left text-sm font-semibold tracking-wide focus:outline-none"
					>Price: High to Low
				</button>	
			</li>
			<li>
				<button
					type="button"
					class="w-full text-left text-sm font-semibold tracking-wide focus:outline-none"
					>Price: Low to High
				</button>
			</li>
		</ul>
	{:else if id == "video"}
		<p>video</p>
	{/if}
</Modal> -->
