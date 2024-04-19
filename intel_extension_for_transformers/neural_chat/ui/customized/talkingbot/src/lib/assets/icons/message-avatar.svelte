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
	import PersonOutlined from "$lib/assets/icons/portrait.svelte";
	import AssistantIcon from "$lib/assets/icons/assistant.svelte";
	import { CollectionType, TalkingPhotoCustom, TalkingTemplateCustom, currentAvaTar } from "$lib/components/talkbot/store";
	import { TalkingTemplateLibrary, TalkingPhotoLibrary } from "$lib/components/talkbot/constant";
	import { MESSAGE_ROLE } from "$lib/components/shared/shared.type";
	export let role: string;

	const map: {[key: number]: {avatar: string}[]} = {
		[CollectionType.Custom]: $TalkingPhotoCustom,
		[CollectionType.Library]: TalkingPhotoLibrary,
		[CollectionType.TemplateLibrary]: TalkingTemplateLibrary,
		[CollectionType.TemplateCustom]: $TalkingTemplateCustom,
	}
	$: src = map[$currentAvaTar.collection][$currentAvaTar.id]?.avatar
</script>

{#if role === MESSAGE_ROLE.USER}
	<PersonOutlined />
{:else if $currentAvaTar}
	<img alt='' {src} class="mx-auto object-cover rounded h-12 w-12 "/>
{:else}
	<AssistantIcon />
{/if}
