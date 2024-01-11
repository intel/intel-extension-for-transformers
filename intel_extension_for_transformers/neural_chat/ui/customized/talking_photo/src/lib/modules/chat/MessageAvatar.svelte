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
