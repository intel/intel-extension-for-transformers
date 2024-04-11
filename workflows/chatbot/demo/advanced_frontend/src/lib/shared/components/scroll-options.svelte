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
	import { MODEL_OPTION } from "$lib/shared/shared.constant";
	export let lists: typeof MODEL_OPTION;
	export let selected = "";
	

	function resetOptionValue() {
		lists.options.forEach((_, idx) => {
			lists.options[idx].value = MODEL_OPTION.options[idx].value
		})
	}
</script>

<div class="h-full">
	<div class="flex flex-col">
		<label class="flex flex-col items-start label">
			<div class="py-2 text-white">
				<span>Options</span>
			</div>
			<select
				class="select w-full bg-transparent text-white border-white"
				bind:value={selected}
				on:change={() => {resetOptionValue()}}
			>
				{#each lists.names as name}
					<option class="bg-transparent text-black" value={name}>
						{name}
					</option>
				{/each}
			</select>
		</label>
	</div>

	<div
		class="shadow border-2 border-transparent"
	>
		<div class="flex flex-col justify-start overflow-hidden">
			{#each lists.options as option}
				<label class="label flex-col items-start bg-transparent text-white text-sm my-2">
					{#if option.type == 'range'}
						<span>{option.label}: {option.value}</span>
						<input
							class="w-full input-range bg-blue-400"
							type="range"
							max={option.maxRange}
							min={option.minRange}
							step={option.step}
							bind:value={option.value}
						/>
					{/if}
				</label>
			{/each}
		</div>
	</div>
</div>
