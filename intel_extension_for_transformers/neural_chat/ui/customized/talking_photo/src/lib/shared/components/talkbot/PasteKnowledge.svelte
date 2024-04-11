<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

<script lang="ts">
	import { Button, Helper, Input, Label, Modal } from "flowbite-svelte";
	import { createEventDispatcher } from "svelte";
	import Paste from "$lib/assets/handle/Paste.svelte";
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
