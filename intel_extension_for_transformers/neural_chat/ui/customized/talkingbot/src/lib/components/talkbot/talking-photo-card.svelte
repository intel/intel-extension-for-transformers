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
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import EditIcon from "./imgs/Edit.svelte";
	import DeleteIcon from "./imgs/Delete.svelte";

	export let name: string;
	export let avatar: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	let dispatch = createEventDispatcher();
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

<div class="relative flex w-28 flex-col items-center">
	<div class="relative">
		<span class="relative block">
			<img
				alt={name}
				src={avatar}
				class="mx-auto h-24 w-24 rounded object-cover hover:border"
			/>
			{#if notLibrary}
				<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
			{/if}
		</span>
	</div>

	<span class="relative mt-2 text-xs text-gray-600  text-ellipsis overflow-hidden whitespace-nowrap">{name}</span>

</div>
