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
	import InlineEditable from "$lib/shared/components/editable/InlineEditable.svelte";
	import Delete from "$lib/assets/image-info/svelte/delete.svelte";
	import { createEventDispatcher } from "svelte";

	const dispatch = createEventDispatcher();

	export let idx: number;
	export let tagKey: string;
	export let tagValue: string;

	
	let prevTagKey = tagKey;
	let prevTagValue = tagValue;

	function handleTagChange(newKey: string, newValue: string) {
		const payload = {
			idx: idx,
			newKey: newKey,
			newValue: newValue,
		};
		dispatch("edit", payload);
	}

	$: if (tagKey !== prevTagKey || tagValue !== prevTagValue) {
		handleTagChange(tagKey, tagValue);
		prevTagKey = tagKey;
		prevTagValue = tagValue;
	}
</script>

<div class="flex flex-wrap items-center gap-2">
	<div
		class="border-custom-2 flex items-center gap-3 rounded-lg border bg-[#f4fbff]"
	>
		<InlineEditable extraClass="text-sm bg-[#f4fbff]" bind:value={tagKey} inputExtraClass="block w-full  text-sm text-gray-900 text-center"/>：
		<InlineEditable extraClass="text-sm bg-[#f4fbff]" bind:value={tagValue} inputExtraClass="block w-full text-sm text-gray-900 text-center" />
		<button
		    class="mr-2"
			on:click={() => {
				dispatch("delete");
			}}><Delete /></button
		>
	</div>
</div>
