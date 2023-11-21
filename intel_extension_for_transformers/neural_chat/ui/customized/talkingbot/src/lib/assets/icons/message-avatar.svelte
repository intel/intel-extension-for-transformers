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
