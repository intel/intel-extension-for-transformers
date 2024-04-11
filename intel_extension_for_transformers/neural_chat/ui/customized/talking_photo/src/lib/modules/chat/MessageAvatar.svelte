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
	import AssistantIcon from "$lib/assets/chat/svelte/Assistant.svelte";
	import PersonOutlined from "$lib/assets/chat/svelte/PersonOutlined.svelte";
	import { TalkingTemplateLibrary } from "$lib/shared/constant/Data";
	import { MessageRole } from "$lib/shared/constant/Interface";
	import { CollectionType, TemplateCustom, currentTemplate } from "$lib/shared/stores/common/Store";
	export let role: MessageRole;

	$: map = {
		[CollectionType.Custom]: $TemplateCustom,
		[CollectionType.Library]: TalkingTemplateLibrary,
	}
	
	$: src = map[$currentTemplate.collection][$currentTemplate.id] ?.avatar;
</script>

{#if role === MessageRole.User}
	{#if $currentTemplate}
		<img alt='' {src} class="mx-auto object-cover rounded h-full w-full "/>
	{:else}
		<PersonOutlined />
	{/if}
{:else}
	<AssistantIcon />
{/if}
