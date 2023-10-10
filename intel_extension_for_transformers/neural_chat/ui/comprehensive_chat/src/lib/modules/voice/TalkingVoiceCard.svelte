<script lang="ts">
	import stopRecordingIcon from "$lib/assets/chat/svg/StopRecording.svg";
	import EditIcon from "$lib/assets/voice/svelte/Edit.svelte";
	import DeleteIcon from "$lib/assets/voice/svelte/Edit.svelte";
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import VoiceButton from "$lib/assets/chat/svelte/VoiceButton.svelte";
	export let name: string;
	export let audio: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	const dispatch = createEventDispatcher();
	let inputEl: HTMLInputElement;

	onMount(() => {
		music = new Audio(audio);
		music.onended = () => {
			music.currentTime = 0;
			play = false;
			music.pause();
		};
		if (needChangeName) {
			changeName();
		}
	});

	afterUpdate(() => {
		if (inputEl) {
			inputEl.focus();
			inputEl.onblur = () => {
				needChangeName = false;
			};
		}
	});

	let play = false;
	let music: HTMLAudioElement;

	function handleAudioPlayer() {
		play = !play;
		if (play) {
			music.play();
		} else {
			music.pause();
		}
	}

	function changeName() {
		needChangeName = true;
	}
</script>

<div
	class="relative flex h-28 w-28 flex-col items-center rounded-xl pt-4 shadow-[0_2px_30px_0_rgba(0,0,0,0.1)]"
>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<div
		class="flex items-center justify-center"
		on:click|stopPropagation={() => {
			handleAudioPlayer();
		}}
	>
		{#if play}
			<img class="my-1 h-11 w-11" src={stopRecordingIcon} alt="" />
		{:else}
			<VoiceButton />
		{/if}
	</div>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	{#if notLibrary}
		<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
	{/if}
	{#if needChangeName}
		<input
			type="text"
			bind:value={name}
			bind:this={inputEl}
			class="mt-2 w-full text-center text-sm text-gray-600 focus-visible:outline-[#ccc] dark:text-gray-400 "
		/>
	{:else}
		<span
			class="relative mt-2 w-9/12 text-sm text-gray-600 dark:text-gray-400 text-ellipsis"
			on:dblclick|capture={changeName}
		>
			{name}
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			{#if notLibrary}
				<span class="absolute -right-2 -top-1"><EditIcon on:changeName={changeName} /></span>
			{/if}
		</span>
	{/if}
</div>
