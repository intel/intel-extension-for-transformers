<script lang="ts">
	import Upload from "$lib/assets/handel/Upload.svelte";
	import { createEventDispatcher } from "svelte";

	const dispatch = createEventDispatcher();

	// let dialog: HTMLDialogElement

	function handleInput(event: Event) {
		const file = (event.target as HTMLInputElement).files![0];
		if (!file) return;

		const reader = new FileReader();
		reader.onloadend = () => {
			if (!reader.result) return;
			const src = reader.result.toString();
			dispatch("upload", { src: src, fileName: file.name });
		};
		reader.readAsDataURL(file);
	}
</script>

<Upload type="audio" />


<input
	id="audio"
	type="file"
	required
	on:change={handleInput}
	accept="audio/*"
/>

<style lang="postcss">
	input[type="file"] {
		display: none;
	}
</style>
