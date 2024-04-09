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
	import Knowledge from "$lib/assets/knowledge/svelte/Knowledge.svelte";
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import EditIcon from "$lib/assets/avatar/svelte/Edit.svelte";
	import DeleteIcon from "$lib/assets/avatar/svelte/Delete.svelte";

	export let name: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	const dispatch = createEventDispatcher();
	let inputEl: HTMLInputElement;

	onMount(() => {
		if (needChangeName) {
			changeName();
		}
	});
	afterUpdate(() => {
		if (inputEl) {
			inputEl.focus();
			inputEl.onblur = () => {
				needChangeName = false;
			};
		}
	});

	function changeName() {
		needChangeName = true;
	}
</script>

<div
	class="flex h-full w-full items-center justify-center rounded-xl pt-1 shadow-[0_2px_30px_0_rgba(0,0,0,0.1)]"
>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<div class="flex w-full flex-col items-center justify-center">
		<Knowledge />
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		{#if notLibrary}
			<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
		{/if}
		<span
			class="relative w-full overflow-hidden text-ellipsis whitespace-nowrap text-xs text-gray-600 truncate"
		>
			{name}
		</span>
	</div>
</div>
