<script lang="ts">
	import Clear from "$lib/assets/chat/svelte/Clear.svelte";
	import { Button, Tooltip } from "flowbite-svelte";
	export let time: string;
	import { createEventDispatcher } from "svelte";

	let dispatch = createEventDispatcher();
	export let msecond_per_token = "0";
	export let first_token_latency = "0";
	let dispatchTime = false;
</script>

<div class="ml-2 flex flex-col">
	<div class="my-1 mr-3 flex items-center justify-end gap-2 space-x-2">
		<div class="ml-2 w-min cursor-pointer" data-state="closed">
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<Button class="border-none bg-transparent p-0 hover:bg-transparent"
			on:click={() => {
				dispatchTime = !dispatchTime;
			}}
			>
				<svg
					t="1712143801854"
					class="icon"
					viewBox="0 0 1024 1024"
					version="1.1"
					xmlns="http://www.w3.org/2000/svg"
					p-id="4259"
					width="20"
					height="20"
					><path
						d="M512.9 63.4C265.4 63.4 64.8 264 64.8 511.5 64.8 759 265.5 959.6 513 959.6c247.4 0 448-200.6 448-448.1C961 264 760.4 63.4 512.9 63.4zM737 560.9H499.1c-5 0-9.9-1-14.3-2.9-13.2-5.6-22.4-18.6-22.4-33.8V286.3c0-20.3 16.4-36.7 36.7-36.7 20.3 0 36.8 16.4 36.8 36.7v201.2h201.2c20.3 0 36.7 16.4 36.7 36.7 0 20.3-16.5 36.7-36.8 36.7z"
						fill="#459df5"
						p-id="4260"
					/></svg
				>
				
			</Button>
			<Tooltip>Time</Tooltip>
		</div>
		<div
			class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
		/>
		<div class="ml-2 w-min cursor-pointer" data-state="closed">
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<Button class="border-none bg-transparent p-0 hover:bg-transparent">
				<svg
					t="1710401975353"
					class="icon"
					viewBox="0 0 1024 1024"
					version="1.1"
					xmlns="http://www.w3.org/2000/svg"
					p-id="4252"
					width="20"
					height="20"
					on:click={() => {
						dispatch("handelClear");
					}}
					><path
						d="M899.1 869.6l-53-305.6H864c14.4 0 26-11.6 26-26V346c0-14.4-11.6-26-26-26H618V138c0-14.4-11.6-26-26-26H432c-14.4 0-26 11.6-26 26v182H160c-14.4 0-26 11.6-26 26v192c0 14.4 11.6 26 26 26h17.9l-53 305.6c-0.3 1.5-0.4 3-0.4 4.4 0 14.4 11.6 26 26 26h723c1.5 0 3-0.1 4.4-0.4 14.2-2.4 23.7-15.9 21.2-30zM204 390h272V182h72v208h272v104H204V390z m468 440V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H416V674c0-4.4-3.6-8-8-8h-48c-4.4 0-8 3.6-8 8v156H202.8l45.1-260H776l45.1 260H672z"
						fill="#1296DB"
						p-id="4253"
					/></svg
				>
			</Button>
			<Tooltip>Clear All</Tooltip>
		</div>
		<div
			class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
		/>
		<div class="ml-2 w-min cursor-pointer" data-state="closed">
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<Button class="border-none bg-transparent p-0 hover:bg-transparent">
				<svg
					t="1710402718399"
					class="icon"
					viewBox="0 0 1024 1024"
					version="1.1"
					xmlns="http://www.w3.org/2000/svg"
					p-id="9675"
					width="20"
					height="20"
					on:click={() => {
						dispatch("handleTop");
					}}
					><path
						d="M512 960c35.2 0 64-28.8 64-64V128c0-35.2-28.8-64-64-64s-64 28.8-64 64v768c0 35.2 28.8 64 64 64z"
						fill="#92C8FF"
						p-id="9676"
					/><path
						d="M301.145 429.344L557.34 173.138c24.881-24.882 24.881-65.599 0-90.482-24.883-24.885-65.602-24.885-90.485 0L210.66 338.862c-24.881 24.882-24.881 65.599 0 90.482 24.883 24.884 65.602 24.884 90.485 0z"
						fill="#409EFF"
						p-id="9677"
					/><path
						d="M722.855 429.344L466.66 173.138c-24.881-24.882-24.881-65.599 0-90.482 24.883-24.885 65.602-24.885 90.485 0L813.34 338.862c24.881 24.882 24.881 65.599 0 90.482-24.883 24.884-65.602 24.884-90.485 0z"
						fill="#409EFF"
						p-id="9678"
					/></svg
				>
			</Button>
			<Tooltip>Top</Tooltip>
		</div>

		<!-- <div
			class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
		/>
		<div class="w-min cursor-pointer" data-state="closed">
			<Clear />
		</div>
		<div class="flex items-center space-x-1 text-xs text-white">
			<p>Time:</p>
			<p>{time}s</p>
		</div> -->
	</div>
	{#if dispatchTime && msecond_per_token !== "0" && first_token_latency !== "0"}
		<div class="text-white opacity-80 flex items-center space-x-1 text-[0.7rem]">
			<strong>First Token Latency: </strong>
			<p>{first_token_latency}</p>
		</div>
		<div
			class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
		/>
		<div class="text-white opacity-80 flex items-center space-x-1 text-[0.7rem]">
			<strong>Average Token Latency: </strong>
			<p>{msecond_per_token}</p>
		</div>
	{/if}
	<div class="ml-2 flex flex-col" />
</div>
