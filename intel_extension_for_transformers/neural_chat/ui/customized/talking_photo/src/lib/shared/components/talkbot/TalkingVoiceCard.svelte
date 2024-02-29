<script lang="ts">
	import stopRecordingIcon from "$lib/assets/chat/svg/StopRecording.svg";
	import EditIcon from "$lib/assets/voice/svelte/Edit.svelte";
	import DeleteIcon from "$lib/assets/voice/svelte/Delete.svelte";
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import VoiceButton from "$lib/assets/chat/svelte/VoiceButton.svelte";
	import StopRecording from "$lib/assets/voice/svelte/StopRecording.svelte";
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
	class="relative flex flex-col justify-center items-center h-full w-full flex-col items-center rounded-xl shadow-[0_2px_30px_0_rgba(0,0,0,0.1)]"
>
	<button
		class="w-7 h-7"
		on:click|stopPropagation={(e) => {
			handleAudioPlayer();
		}}
	>
		{#if play}
			<StopRecording extraClass="h-7 w-7"/>
		{:else}
			<VoiceButton />
		{/if}
	</button>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	{#if notLibrary}
		<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
	{/if}
	{#if needChangeName}
		<input
			type="text"
			bind:value={name}
			bind:this={inputEl}
			class="w-4/5 p-0 text-center text-sm text-gray-600 focus-visible:outline-[#ccc] dark:text-gray-400 "
		/>
	{:else}
		<span
			class="relative text-xs text-gray-600 dark:text-gray-400 text-ellipsis truncate"
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
