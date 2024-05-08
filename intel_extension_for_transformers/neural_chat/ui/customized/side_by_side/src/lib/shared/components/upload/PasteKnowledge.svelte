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

	const dispatch = createEventDispatcher();
	let formModal = false;
	let urlValue = "";

	function handelPasteURL() {
		const pasteUrlList = urlValue.split(";").map((url) => url.trim());
		dispatch("paste", { pasteUrlList });
		formModal = false;
	}
</script>

<Label class="space-y-1">
	<div class="grid grid-cols-3">
		<Input
			class="col-span-2 rounded-none rounded-l-lg focus:border-blue-700 focus:ring-blue-700"
			type="text"
			name="text"
			placeholder="URL"
			bind:value={urlValue}
		/>
		<Button
			type="submit"
			class="w-full rounded-none rounded-r-lg bg-blue-700"
			on:click={() => handelPasteURL()}>Confirm</Button
		>
	</div>

	<Helper>Use semicolons (;) to separate multiple URLs.</Helper>
</Label>
