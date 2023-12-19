<script>
	import {
		Button,
		Toggle,
		Tabs,
		TabItem,
		Select,
		Modal,
	} from "flowbite-svelte";
	import SettingIcon from "$lib/assets/chat/svelte/Setting.svelte";
	import DataIcon from "$lib/assets/chat/svelte/Data.svelte";
	import Timer from "$lib/modules/settings/Timer.svelte";
	import { onCountdownEnd } from "$lib/network/settings/Network";
	import { LOCAL_STORAGE_KEY } from "$lib/shared/constant/Interface";
	import { getNotificationsContext } from "svelte-notifications";
	import { createEventDispatcher } from "svelte";

	const { addNotification } = getNotificationsContext();
	import {
		countDown,
		currentMode,
		ifStoreMsg,
		resetControl,
	} from "$lib/shared/stores/common/Store";
	import { Icon } from "flowbite-svelte-icons";
	import { formatTime } from "$lib/shared/Utils";

	$: currentCountdown = $countDown;
	const dispatch = createEventDispatcher();
	let popupModal = false;

	// let selected;
	let countries = [
		{
			value: "Default, normal browsing with data saved.",
			name: "Standard",
		},
		{
			value: " Enhanced privacy, blocking trackers and limiting cookies.",
			name: "Strict",
		},
		{
			value: "Browsing without saving history or cookies.",
			name: "Anonymous",
		},
	];

	function initMessage() {
		localStorage.setItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY, "[]");

		localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY);
		addNotification({
			text: "Clear successfully",
			position: "bottom-center",
			type: "success",
			removeAfter: 1000,
		});
	}

	function deleteAccount() {
		localStorage.setItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY, "[]");
		currentMode.set("Text");

		resetControl.set(true);
		addNotification({
			text: "Deleted successfully",
			position: "bottom-center",
			type: "success",
			removeAfter: 1000,
		});
	}

	function handleToggle(e) {
		ifStoreMsg.set(e.target.checked);
	}
</script>

<div class="container mx-auto px-4 py-8 sm:px-6 sm:py-20 lg:px-8">
	<h2 class="text-center text-2xl font-bold">Settings</h2>
	<div
		class="relative mx-auto max-w-xl rounded-3xl px-12 px-6 py-6 shadow-[20px_34px_74px_0_#15156107] max-sm:px-2"
	>
		<Tabs
			style="full"
			defaultClass="flex rounded-lg divide-x divide-gray-200 shadow "
			tabStype="full"
		>
			<TabItem class="w-full" open>
				<span
					slot="title"
					class="flex items-center justify-center gap-2 font-bold"
					><SettingIcon />General</span
				>
				<div class="">
					<div class="flex w-full flex-row items-center justify-between py-4">
						<label for="save-chat" class="block w-full font-bold md:text-sm"
							>Save Conversation</label
						>

						{#if currentCountdown > 0}
							<p>{formatTime(currentCountdown)}</p>
						{/if}
					</div>
					<div class="">
						<div class="flex w-full flex-row items-center justify-between py-4">
							<label for="save-chat" class="block w-full font-bold md:text-sm"
								>Clear all chats</label
							>
							<button
								type="button"
								class="mb-2 mr-2 rounded-lg bg-red-700 px-12 py-2 text-sm font-medium text-white focus:outline-none"
								on:click={() => initMessage()}>Clear</button
							>
						</div>
					</div>
				</div></TabItem
			>
			<TabItem class="w-full">
				<span
					slot="title"
					class="flex items-center justify-center gap-2 font-bold"
					><DataIcon />Data controls</span
				>
				<div>
					<div class="flex justify-between">
						<h2 class="font-bold">Chat history & training</h2>
						<Toggle
							color="blue"
							checked={$ifStoreMsg}
							on:change={handleToggle}
						/>
					</div>

					<div class="mt-4">
						<p class="text-sm text-gray-500">
							Your data, including personal information and uploaded images, can
							be securely stored in our database to improve your user
							experience. However, we prioritize your privacy, and we kindly
							request your permission to retain this data for future access,
							giving you control over your information.
						</p>
					</div>
				</div>

				<div class="mt-4 space-y-1">
					<div class="flex w-full flex-row items-center justify-between">
						<label for="save-chat" class="block w-full font-bold md:text-sm"
							>Delete account</label
						>
						<button
							type="button"
							class="mb-2 mr-2 rounded-lg bg-red-700 px-5 py-2 text-sm font-medium text-white focus:outline-none"
							on:click={() => {
								popupModal = true;
							}}>Delete</button
						>
					</div>
					<p class="text-sm text-gray-500">
						Note: Both chat history and account will be cleared.
					</p>
				</div>
			</TabItem>
		</Tabs>
	</div>
</div>

<Modal bind:open={popupModal} size="xs" autoclose>
	<div class="text-center">
		<Icon
			name="exclamation-circle-outline"
			class="mx-auto mb-4 h-12 w-12 text-gray-400"
		/>
		<h3 class="mb-5 text-lg font-normal text-gray-500 dark:text-gray-400">
			Confirm delete this Account?
		</h3>
		<Button
			color="red"
			class="mr-2"
			on:click={() => {
				deleteAccount();
			}}>Yes, I'm sure</Button
		>
		<Button color="alternative">No, cancel</Button>
	</div>
</Modal>
