<script lang="ts">
	import Upload from "$lib/assets/handel/Upload.svelte";
	import { createEventDispatcher } from "svelte";

	const dispatch = createEventDispatcher();

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

<div class="aspect-square w-full">
	<Upload type="file" />

	<input
		id="file"
		type="file"
		accept=".doc, .docx, .pdf, .xls, .xlsx, .txt, .json, application/pdf, application/msword, application/vnd.ms-excel, text/plain, application/json"
		required
		on:change={handleInput}
	/>
</div>

<style lang="postcss">
	input[type="file"] {
		display: none;
	}
</style>
