<script lang="ts">
	import { Alert } from "flowbite-svelte";
	import { onMount } from "svelte";
	import ArrowDown from "$lib/assets/layout/ArrowDown.svelte";
	import ArrowUp from "$lib/assets/layout/ArrowUp.svelte";
	import { trafficHint } from "$lib/shared/Utils";

	const interval = 5000;
	const closeAlert = () => {
		isAlertOpen = false;
	};

	let isAlertOpen = true;
	let content = "";

	const updateContent = async () => {
		const result = await trafficHint();
		if (result) {
			let count = result;
			if (count > 5) {
				closeAfterDelay();
				if (count < 10) {
					content = count + ` requests in the queue. Please wait a moment!`;
				} else if (count >= 10 && count < 20) {
					content = `10+  requests in the queue. Please wait a moment!`;
				} else if (count >= 20) {
					content = `20+  requests in the queue. Please wait a moment!`;
				}
			} else {
				content = "";
			}
		} else {
			content = "";
		}

		setTimeout(updateContent, interval);
	};

	const closeAfterDelay = () => {
		setTimeout(() => {
			isAlertOpen = false;
		}, 5000);
	};

	const openAlert = () => {
		isAlertOpen = !isAlertOpen;
	};

	onMount(() => {
		setTimeout(updateContent, interval);
	});
</script>

{#if content != ""}
	{#if isAlertOpen}
		<Alert color="yellow" class="-mb-3 w-full border-t-4 z-20" on:close={closeAlert}>
			{content}
		</Alert>
		<button
			on:click={openAlert}
			class="absolute right-5 top-6 z-20 w-2 bg-yellow-50 z-20 "><ArrowDown /></button
		>
	{:else}
		<button
			on:click={openAlert}
			class="absolute left-0 top-2 z-20 rounded border p-1.5 shadow z-20"
			><ArrowUp /></button
		>
	{/if}
{/if}
