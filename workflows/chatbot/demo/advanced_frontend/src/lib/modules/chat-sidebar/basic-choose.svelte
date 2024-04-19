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
	import { DOMAIN_LIST } from "$lib/shared/shared.constant";
	
	export let domain_idx: number;
</script>

<div class="grid gap-6 grid-cols-1 text-title px-2 mt-4">
	{#each DOMAIN_LIST as domain, idx}
		<button
			class="w-full text-gray-300 hover:bg-gray-500 flex items-center px-2 py-3 text-sm font-medium rounded-md mb-2 shadow-sm ring-1 ring-inset ring-gray-400 text-white"
			class:bg-[#224c75]={idx === domain_idx}
			class:btn-disabled={idx > 1}
			on:click={() => (domain_idx = idx)}
		>
			<img src={domain.svg} class="w-5 mx-2" alt="" />
			{domain.title}
		</button>
	{/each}
</div>
