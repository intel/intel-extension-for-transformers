<script lang="ts">
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import ChatMessage from "$lib/modules/chat/ChatMessage.svelte";
	import "driver.js/dist/driver.css";
	import "$lib/assets/layout/css/driver.css";
	import { scrollToBottom } from "$lib/shared/Utils";
	import { onMount } from "svelte";
	import { createEventDispatcher } from "svelte";

	let dispatch = createEventDispatcher();

	let scrollToDiv: HTMLDivElement;

	export let chatMessages: any;
	export let label: string;
	export let scrollName: string;
	console.log(chatMessages, label);
</script>

<div
	id="custom-controls-gallery"
	class="custom-controls-gallery relative mb-8 h-0 w-full grow px-2 {scrollName}"
	data-carousel="slide"
>
	<!-- Carousel wrapper -->
	<!-- Display current item -->
	<Scrollbar
		classLayout="flex flex-col gap-5 h-full"
		className=" h-0 w-full grow px-2 mt-3 ml-10"
	>
		{#each chatMessages as message, i}
			<ChatMessage
				on:scrollTop={() => dispatch("ExternalTop")}
				on:handelExternalClear={() => dispatch("ExternalClear")}
				msg={message}
				time={i === 0 || (message.time > 0 && message.time < 100)
					? message.time
					: ""}
			/>
		{/each}
	</Scrollbar>
	<!-- Loading text -->
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
