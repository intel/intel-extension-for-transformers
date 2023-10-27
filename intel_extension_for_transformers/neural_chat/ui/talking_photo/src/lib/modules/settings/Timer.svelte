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
			console.log('resetControl');
			
			resetCountdown();
			resetControl.set(false);
		}
	}

	async function resetCountdown() {
		console.log("coming");

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
		console.log("storedCountdown", storedCountdown);

		if (storedCountdown) {
			countdown = parseInt(storedCountdown, 10);
		} else {
			console.log("storedCountdown error");
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