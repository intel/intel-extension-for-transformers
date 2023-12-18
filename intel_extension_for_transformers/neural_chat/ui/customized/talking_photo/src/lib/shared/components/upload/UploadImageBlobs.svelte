<script lang="ts">
	import { createEventDispatcher } from "svelte";
	import upload from "$lib/assets/home/imgs/upload.png";
	import Upload from "$lib/assets/handel/Upload.svelte";


	const dispatch = createEventDispatcher();

	function handleInput(event: Event) {
		const files = (event.target as HTMLInputElement).files;

		if (!files) return;

		dispatch("upload", { blobs: files });
		// for (let i = 0; i < files.length; ++i) {
		// 	const reader = new FileReader();
		// 	reader.onloadend = () => {
		// 		if (!reader.result) return;
		// 		const src = reader.result.toString();
		// 		dispatch("upload", { src: src, fileName: files[i].name });
		// 	};
		// 	reader.readAsDataURL(files[i]);
		// }
	}
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div class="relative" on:click|capture|nonpassive|stopPropagation={() => {}}>
	<label for="image" class="h-full w-full cursor-pointer text-center">
		<slot>
			<Upload type="image" />
		</slot>
	</label>
	<input
		id="image"
		type="file"
		required
		on:change={handleInput}
		accept="image/*"
		multiple
	/>
</div>

<style lang="postcss">
	input[type="file"] {
		display: none;
	}
</style>
