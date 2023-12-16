<script lang="ts">
	import { Button, Helper, Input, Label, Modal } from "flowbite-svelte";
	import { createEventDispatcher } from "svelte";
	import Paste from "$lib/assets/handel/Paste.svelte";
	const dispatch = createEventDispatcher();
	let formModal = false;
	let urlValue = "";

	function handelPasteURL() {
		const pasteUrlList = urlValue.split(';').map(url => url.trim());
		dispatch("paste", { pasteUrlList });
		formModal = false;
	}
</script>

<div class="aspect-square w-full sm:w-[5rem] sm:h-[5rem] ">
	<label for="file" class="h-full w-full text-center">
			<button
				on:click={() => (formModal = true)}
				class="flex h-full w-full cursor-pointer flex-col justify-center rounded-md"
			>
				<Paste />
			</button>
	</label>
</div>

<Modal bind:open={formModal} size="xs" autoclose={false} class="w-full">
	<Label class="space-y-2">
		<span>Paste URL</span>
		<Input
			type="text"
			name="text"
			placeholder="URL"
			bind:value={urlValue}
		/>
		<Helper>Use semicolons (;) to separate multiple URLs.</Helper>
	</Label>

	<Button type="submit" class="w-full" on:click={() => handelPasteURL()}
		>Confirm</Button
	>
</Modal>
