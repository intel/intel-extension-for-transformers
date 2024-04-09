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

<script>
	import { LOCAL_STORAGE_KEY } from "$lib/shared/constant/Interface";
	import { onMount } from "svelte";
	import { getNotificationsContext } from "svelte-notifications";
	import { onCountdownEnd } from "$lib/network/settings/Network";
	import { goto } from "$app/navigation";
	import { countDown, resetControl } from "$lib/shared/stores/common/Store";
	const { addNotification } = getNotificationsContext();

	let countdown = $countDown;
	let canShowNotification = true;

	const initialCountdownValue = 1800;

	$: {
		if ($resetControl) {			
			resetCountdown();
			resetControl.set(false);
		}
	}

	async function resetCountdown() {
		if (canShowNotification && !$resetControl) {
			addNotification({
				text: "session timeout",
				position: "bottom-center",
				type: "error",
				removeAfter: 1000,
			});
			canShowNotification = false;
			setTimeout(() => {
				canShowNotification = true;
			}, 2000);
		}
		countdown = initialCountdownValue;
		localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_TIME_KEY);
		localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY);
		await onCountdownEnd();
		goto("/");
	}

	onMount(async () => {

		await Promise.resolve();

		const storedCountdown = localStorage.getItem(
			LOCAL_STORAGE_KEY.STORAGE_TIME_KEY
		);
		if (storedCountdown) {
			countdown = parseInt(storedCountdown, 10);
		} else {
			countdown = initialCountdownValue;
		}

		const interval = setInterval(() => {
			countdown--;
			countDown.update((newValue) => {
				return countdown;
			});

			if (countdown <= 0) {
				resetCountdown();
			}

			localStorage.setItem(
				LOCAL_STORAGE_KEY.STORAGE_TIME_KEY,
				countdown.toString()
			);
		}, 1000);

		return () => {
			clearInterval(interval);
		};
	});
</script>

<div class="hidden">
	{#if countdown > 0}
		<h1>{countdown}</h1>

		<button on:click={resetCountdown} 
			>reset</button
		>
	{/if}
</div>