<script lang="ts">
    import uploadFile from '$lib/assets/voice/svg/uploadFile.svg';
	import KnowledgeBaseChoise from '$lib/modules/customize/KnowledgeBaseChoise.svelte';

	let showModal = false;
</script>

<button class="aspect-square w-full flex flex-col items-center justify-center shadow-[0_2px_30px_0_rgba(0,0,0,0.1)] rounded-xl text-center  sm:w-[5rem] sm:h-[5rem] " on:click={() => showModal = true}>
	<img class="h-10" src={uploadFile} alt=""/>
	<p class="text-xs text-gray-500">Create</p>
</button>

<KnowledgeBaseChoise bind:showModal on:paste on:upload></KnowledgeBaseChoise>
