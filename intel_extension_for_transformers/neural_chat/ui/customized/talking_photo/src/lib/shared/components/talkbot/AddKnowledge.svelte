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
    import uploadFile from '$lib/assets/voice/svg/uploadFile.svg';
	import KnowledgeBaseChoise from '$lib/modules/customize/KnowledgeBaseChoise.svelte';

	let showModal = false;
</script>

<button class="aspect-square w-full flex flex-col items-center justify-center shadow-[0_2px_30px_0_rgba(0,0,0,0.1)] rounded-xl text-center  sm:w-[5rem] sm:h-[5rem] " on:click={() => showModal = true}>
	<img class="h-10" src={uploadFile} alt=""/>
	<p class="text-xs text-gray-500">Create</p>
</button>

<KnowledgeBaseChoise bind:showModal on:paste on:upload></KnowledgeBaseChoise>
