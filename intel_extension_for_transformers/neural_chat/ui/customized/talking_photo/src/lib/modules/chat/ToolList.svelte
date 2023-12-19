<script lang="ts">
	import {
		currentMode,
		imageList,
		knowledgeAccess,
		photoMode,
		popupModal,
		videoMode,
	} from "$lib/shared/stores/common/Store";
	import { createEventDispatcher } from "svelte";
	import { Button, Dropdown, Popover, Radio, Tooltip } from "flowbite-svelte";
	import Video from "$lib/assets/chat/svelte/Video.svelte";
	import Text from "$lib/assets/chat/svelte/Text.svelte";
	import ColorImg from "$lib/assets/chat/svelte/ColorImg.svelte";
	import KnowledgeAccess from "$lib/assets/chat/svelte/KnowledgeAccess.svelte";
	import TextIcon from "$lib/assets/chat/svelte/TextIcon.svelte";
	import SearchPhoto from "$lib/assets/chat/svelte/SearchPhoto.svelte";
	import PhotoTransfer from "$lib/assets/chat/svelte/PhotoTransfer.svelte";
	import VideoChat from "$lib/assets/chat/svelte/VideoChat.svelte";
	import VideoIcon from "$lib/assets/chat/svelte/VideoIcon.svelte";

	const dispatch = createEventDispatcher();
	let photoChecked = "photoChat";
	const toolStyle = {
		init: `group mx-1 inline-flex flex-col items-center justify-center rounded-full rounded-r-full bg-white p-1 px-2 hover:bg-gray-50`,
		inactive: ``,
		active: `outline-none ring-2 ring-indigo-500 ring-offset-2 ring-offset-indigo-200`,
	};
	const hintStyle = {
		init: `group mx-1 inline-flex flex-col items-center justify-center rounded-full rounded-r-full bg-white p-1 px-2 hover:bg-gray-50`,
		inactive: ``,
		active: `outline-none ring-2 ring-purple-500 ring-offset-2 ring-offset-purple-200`,
	};
	$: {
		if (photoChecked) {
			photoMode.set(photoChecked);
		}
	}

	function exchangeMode(mode: string) {
		currentMode.set(mode);
	}
	let type = "";
</script>

<div class="m-1 flex inline-flex p-1">
	<div class="rounded-full border border-gray-200 bg-[#f4f5f7] px-2 py-1">
		<div class="mx-auto flex h-full">
			<button
				type="button"
				id="type-Chat with Text"
				class={`${toolStyle.init}  ${
					$currentMode === "Text" ? toolStyle.active : toolStyle.inactive
				}`}
				on:click={() => {
					exchangeMode("Text");
					dispatch("showPrompt", false);
				}}
			>
				<Text />
			</button>

			
			<button
				type="button"
				id="type-Upload Photos"
				class={`${toolStyle.init}  ${
					$currentMode === "Search" ? toolStyle.active : toolStyle.inactive
				}`}
				on:click={() => {
					popupModal.set(true);
					exchangeMode("Search");
					dispatch("showPrompt", true);
				}}
			>
				<SearchPhoto />
			</button>

			<button
				type="button"
				id="type-Stylize Photos"
				class={`${toolStyle.init}  ${
					$currentMode === "Photo" ? toolStyle.active : toolStyle.inactive
				}`}
				on:click={() => {
					popupModal.set(true);
					dispatch("showPrompt", true);
					exchangeMode("Photo");
				}}
			>
				<PhotoTransfer />
			</button>

			<button
				type="button"
				id="type-Create Talking Avatar"
				class={`${toolStyle.init}  ${
					$currentMode === "Video" && $videoMode === "input"
						? toolStyle.active
						: toolStyle.inactive
				}`}
				on:click={() => {
					popupModal.set(true);
					exchangeMode("Video");
					videoMode.set("input");
					dispatch("showPrompt", false);
				}}
			>
				<Video />
			</button>

			<button
				type="button"
				id="type-Chat with Avatar"
				class={`${toolStyle.init}  ${
					$currentMode === "Video" && $videoMode === "output"
						? toolStyle.active
						: toolStyle.inactive
				}`}
				on:click={() => {
					popupModal.set(true);
					exchangeMode("Video");
					videoMode.set("output");
					dispatch("showPrompt", false);
				}}
			>
				<VideoChat />
			</button>
			<Tooltip
				placement="top"
				color={"yellow"}
				{type}
				triggeredBy="[id^='type-']"
				on:show={(ev) => (type = ev.target.id.split("-")[1])}
				class="px-2 py-1 text-[0.7rem]">{type}</Tooltip
			>

			<!-- <Dropdown
				placement="right"
				triggeredBy="#videoChoose"
				class=" max-w-sm divide-y divide-gray-100 rounded p-2 shadow"
			>
				<div class="flex flex-col gap-3">
					<button
						id="type-generate video"
						on:click={() => {
							videoMode.set("input");
						}}
						class={`${hintStyle.init}  ${
							$videoMode === "input" ? hintStyle.active : hintStyle.inactive
						}`}
					>
						<Video />
					</button>

					<button
						id="type-video chat"
						on:click={() => {
							videoMode.set("output");
						}}
						class={`${hintStyle.init}  ${
							$videoMode === "output" ? hintStyle.active : hintStyle.inactive
						}`}
					>
						<VideoChat />
					</button>
					<Dropdown
						class="inline-flex w-[6rem]  items-center justify-center 
						rounded-lg bg-gray-400 p-1 text-center text-[0.7rem] text-white"
						{type}
						placement="right"
						triggeredBy="[id^='type-']"
						on:show={(ev) => {
							type = ev.target.id.split("-")[1];
						}}>{type}</Dropdown
					>
				</div>
			</Dropdown> -->
		</div>
	</div>
</div>
