<script lang="ts">
    import { Button, Helper, Input, Label, ButtonGroup, GradientButton, Modal } from "flowbite-svelte";
    import { createEventDispatcher } from "svelte";
    export let showModal: boolean

    const dispatch = createEventDispatcher();
    let uploadInput: HTMLInputElement

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

	let formModal = false;
	let urlValue = "";

	function handelPasteURL() {
		const pasteUrlList = urlValue.split(';').map(url => url.trim());
		dispatch("paste", { pasteUrlList });
		formModal = false;
	}
</script>

<Modal bind:open={showModal} class="text-center" size="sm">
    <ButtonGroup class="space-x-px">
        <GradientButton on:click={() => uploadInput.click()} class="w-20" color="purpleToBlue">Upload</GradientButton>
        <GradientButton on:click={() => formModal = true} class="w-20" color="greenToBlue">URL</GradientButton>
    </ButtonGroup>
</Modal>

<input
    bind:this={uploadInput}
    type="file"
    accept=".doc, .docx, .pdf, .xls, .xlsx, .txt, .json, application/pdf, application/msword, application/vnd.ms-excel, text/plain, application/json"
    required
    on:change={handleInput}
/>

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

<style lang="postcss">
	input[type="file"] {
		display: none;
	}
</style>