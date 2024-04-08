<script lang="ts">
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import ChatMessage from "$lib/modules/chat/ChatMessage.svelte";
	import "driver.js/dist/driver.css";
	import "$lib/assets/layout/css/driver.css";
	import Previous from "$lib/assets/upload/previous.svelte";
	import Next from "$lib/assets/upload/next.svelte";
	import { scrollToBottom } from "$lib/shared/Utils";
	import { onMount } from "svelte";

	let scrollToDiv: HTMLDivElement;

	export let items;
	export let label: string;
	export let scrollName: string;	

	onMount(async () => {
		scrollToDiv = document
			.querySelector(scrollName)
			?.querySelector(".svlr-viewport")!;
		console.log(
			"scrollToDiv",
			scrollName,
			document,
			document.querySelector("chat-scrollbar1")
		);
	});
	// gallery
	let currentIndex = 0;

	function nextItem() {
		currentIndex = (currentIndex + 1) % items.length;
		console.log("nextItem", currentIndex);
	}

	function prevItem() {
		currentIndex = (currentIndex - 1 + items.length) % items.length;
		console.log("prevItem", currentIndex);
	}

	$: currentItem = items[currentIndex];

	$: {
		if (items) {
			scrollToBottom(scrollToDiv);
		}
	}
	// gallery
</script>

<div
	id="custom-controls-gallery"
	class="relative mb-8 h-0 w-full w-full grow px-2 {scrollName}"
	data-carousel="slide"
>
	<!-- Carousel wrapper -->
	<!-- Display current item -->
	{#if currentItem}
		<Scrollbar
			classLayout="flex flex-col gap-5"
			className="  h-0 w-full grow px-2 mt-3 ml-10"
		>
			{#each currentItem.content as message, i}
				<ChatMessage msg={message} />
			{/each}
		</Scrollbar>
		<!-- Loading text -->
	{/if}

	<div class="radius absolute left-0 p-2">
		<!-- Display end to end time -->
		<label for="" class="mr-2 text-xs font-bold text-blue-700">{label} </label>
	</div>
	{#if currentItem.time !== "0s"}
		<div class="radius absolute right-0 p-2">
			<!-- Display end to end time -->
			<label for="" class="mr-2 text-xs font-bold text-blue-700"
				>End to End Time:
			</label>
			<label for="" class="text-xs">{currentItem.time}</label>
		</div>
	{/if}
	<div class="flex items-center justify-between">
		<div class="justify-left ml-2 flex items-center">
			<!-- Previous button -->
			<button
				type="button"
				class="group absolute start-0 top-0 z-30 flex h-full
									cursor-pointer items-center justify-center
									focus:outline-none"
				on:click={prevItem}
			>
				<span
					class="group-focus:ring-gray dark:group-hover:bg-[#000]-800/60 dark:group-focus:ring-[#000]-800/70 inline-flex h-7
										 w-7 items-center justify-center
										 rounded-full bg-[#000]/10
										 group-hover:bg-[#000]/50 group-focus:bg-[#000]/50
										 group-focus:outline-none
										 group-focus:ring-4 dark:bg-gray-800/30"
				>
					<Previous />
					<span class="sr-only">Previous</span>
				</span>
			</button>
			<!-- Next button -->

			<button
				type="button"
				class="group absolute end-0 top-0 z-30 flex h-full cursor-pointer items-center justify-center focus:outline-none"
				on:click={nextItem}
			>
				<span
					class="group-focus:ring-gray dark:group-hover:bg-[#000]-800/60 dark:group-focus:ring-[#000]-800/70 inline-flex h-7
									w-7 items-center justify-center
									rounded-full bg-[#000]/10
									group-hover:bg-[#000]/50 group-focus:bg-[#000]/50
									group-focus:outline-none
									group-focus:ring-4 dark:bg-gray-800/30"
				>
					<Next />
					<span class="sr-only">Next</span>
				</span>
			</button>
		</div>
	</div>
</div>

<style>
	.row::-webkit-scrollbar {
		display: none;
	}

	.row {
		scrollbar-width: none;
	}

	.row {
		-ms-overflow-style: none;
	}
</style>
