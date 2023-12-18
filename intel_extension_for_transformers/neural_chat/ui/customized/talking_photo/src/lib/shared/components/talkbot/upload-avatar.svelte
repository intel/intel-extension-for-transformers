<script lang="ts">
	import Upload from "$lib/assets/handel/Upload.svelte";
	import { createEventDispatcher } from "svelte";

	const dispatch = createEventDispatcher();

	function handleInput(event: Event) {
		const file = (event.target as HTMLInputElement).files![0];

		if (!file) return;
		const reader = new FileReader();

		var img = new Image();
		img.onload = () => {
			if (img.width && img.height) {
				const src = reader.result!.toString();
				dispatch("upload", { src: src, fileName: file.name });
			} else {
				dispatch("error")
			}
		};
		img.onerror = function() {
			dispatch("error")
		};

		reader.onloadend = () => {
			if (!reader.result) return;
			img.src = reader.result as string;
		};
		reader.readAsDataURL(file);
	}
</script>

<div class="m-auto">
	<Upload type="image" />

	<input
		id="image"
		type="file"
		required
		on:change={handleInput}
		accept="image/*"
	/>
</div>

<style lang="postcss">
	input[type="file"] {
		display: none;
	}
</style>
