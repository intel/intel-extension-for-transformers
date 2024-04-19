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

<script>
	import KnowledgeAccess from "$lib/assets/chat/svelte/KnowledgeAccess.svelte";
	import { TalkingTemplateLibrary } from "$lib/shared/constant/Data";
	import {
		CollectionType,
		TemplateCustom,
		currentTemplate,
	} from "$lib/shared/stores/common/Store";

	$: knowledge =
		$currentTemplate.collection === CollectionType.Custom
			? $TemplateCustom[$currentTemplate.id].knowledge
			: TalkingTemplateLibrary[$currentTemplate.id].knowledge;

	$: knowledge_name =
		$currentTemplate.collection === CollectionType.Custom
			? $TemplateCustom[$currentTemplate.id].knowledge_name
			: TalkingTemplateLibrary[$currentTemplate.id].knowledge_name;
</script>

<div class="flex w-full flex-row items-start gap-1.5 pl-1">
	{#if knowledge === "default" && knowledge_name === "default"}
		<span></span>
	{:else}
		<KnowledgeAccess />

		<span class="text-xs text-[#611fec]">knowledge</span>
		<label class="relative inline-flex cursor-pointer items-center">
			<input type="checkbox" value="" class="peer sr-only" checked on:change />
			<div
				class="peer h-4 w-7 rounded-full bg-gray-200 after:absolute after:left-[1px] after:top-[2px] after:h-3 after:w-3 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-blue-600 peer-checked:after:translate-x-full peer-checked:after:border-white peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:border-gray-600 dark:bg-gray-700 dark:peer-focus:ring-blue-800"
			/>
		</label>
	{/if}
</div>
