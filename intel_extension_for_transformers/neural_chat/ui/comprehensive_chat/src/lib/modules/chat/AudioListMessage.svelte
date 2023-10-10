<script lang="ts">
	import ChatAudio from "$lib/modules/chat/ChatAudio.svelte";
	import TranslateIcon from "$lib/assets/chat/svelte/TranslateIcon.svelte";
	import { Spinner } from "flowbite-svelte";
	import { fetchAudioText } from "$lib/network/chat/Network";

    export let content: string[]
    
    let playIdx = 0;
	let autoPlay = content.length > 0 && content[content.length - 1] !== "done"
	let showTranslateText = false;
	let imgPromise: Promise<any>;

    function handlePlayEnded() {
		playIdx++;
		autoPlay = true;
		if (playIdx < content.length && content[playIdx] === "done") {
			playIdx = 0;
			autoPlay = false;
		}
	}

	async function translateToText(audio: string) {
		console.log(showTranslateText);

		if (showTranslateText == true) {
			showTranslateText = false;
			return;
		}

		imgPromise = (async () => {
			const blob = await fetch(audio).then((r) => r.blob());
			let response = await fetchAudioText(blob);
			return response.asr_result;
		})();

		showTranslateText = true;
	}
</script>

<ChatAudio src={content[0]} {autoPlay} on:ended={handlePlayEnded} />
<div class="absolute -top-5 right-0 hidden h-5 group-hover:flex">
    <button
        class="opacity-40"
        on:click={() => {
            translateToText(content[0]);
        }}
        class:opacity-100={showTranslateText}
    >
        <TranslateIcon />
    </button>
</div>
{#if showTranslateText}
    {#await imgPromise}
        <Spinner size='4' color="gray" />
    {:then translateText}
        <p class="whitespace-pre-line text-sm">{translateText}</p>
    {/await}
{/if}