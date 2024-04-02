<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

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
