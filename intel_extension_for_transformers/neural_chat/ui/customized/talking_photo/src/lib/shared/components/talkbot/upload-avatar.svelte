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
	import Upload from "$lib/assets/handle/Upload.svelte";
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
