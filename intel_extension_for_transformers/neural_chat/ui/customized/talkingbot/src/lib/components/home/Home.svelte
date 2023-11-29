<script lang="ts">
	import BotTemplateCard from "$lib/components/talkbot/bot-template-card.svelte";
	import TalkingPhotoCard from "$lib/components/talkbot/talking-photo-card.svelte";
	import EditIcon from "$lib/assets/icons/Edit.svelte";
	import TalkingVoiceCard from "$lib/components/talkbot/talking-voice-card.svelte";
	import TalkingKnowledgeCard from "$lib/components/talkbot/talking-knowledge-card.svelte";
	import { goto } from "$app/navigation";
	import {
		TalkingTemplateLibrary,
		TalkingPhotoLibrary,
		TalkingVoiceLibrary,
		TalkingKnowledgeLibrary,
		tabList
	} from "../talkbot/constant";
	import {
		CollectionType,
		TalkingPhotoCustom,
		TalkingVoiceCustom,
		TalkingTemplateCustom,
		currentAvaTar,
		currentVoice,
		TalkingKnowledgeCustom,
		currentKnowledge,
	} from "../talkbot/store";

	let mode = "template";

	function tabMode(type: string | undefined) {
		if (!type) {
			goto("/chat");
		} else {
			mode = type;
		}
	}

	console.log('8/3/2023 --- CollectionType', CollectionType, CollectionType.Custom)

	$: allPhotos = [...$TalkingPhotoCustom, ...TalkingPhotoLibrary];
	$: allVoices = [...$TalkingVoiceCustom, ...TalkingVoiceLibrary];
	$: allKnowledges = [...$TalkingKnowledgeCustom, ...TalkingKnowledgeLibrary];
	$: allTemplates = [...$TalkingTemplateCustom, ...TalkingTemplateLibrary];
</script>

<div class="mx-20 -mt-5 flex h-full">
	<div class="flex w-full flex-col gap-5">
		<div
			class="flex-wrap items-center justify-center gap-8 text-center sm:flex"
		>
			{#each tabList as tab}
				<!-- svelte-ignore a11y-click-events-have-key-events -->
				<button
					type="button"
					class="w-full rounded-lg px-4 py-4 shadow-lg transition-all hover:bg-gray-100  sm:w-1/2 md:mt-12 md:w-1/2 lg:mt-16 lg:w-1/4 {mode ===
					tab.type
						? 'bg-white outline-none ring-2 ring-blue-500 ring-offset-2'
						: 'bg-gray-100 '} "
					on:click={() => tabMode(tab.type)}
				>
					<div class="flex-shrink-0">
						<div
							class="mx-auto flex h-12 w-12 items-center justify-center rounded-md bg-blue-600 text-white"
						>
							<EditIcon />
						</div>
					</div>
					<!-- svelte-ignore a11y-missing-content -->
					<h3
						class="py-4 text-2xl font-semibold text-gray-700  sm:text-xl"
					>
						{tab.title}
					</h3>
				</button>
			{/each}
		</div>
		{#if mode === "template"}
			<div>
				<div class="flex justify-between">
					<h3 class="mb-6 text-xl font-medium text-[#051F61]">Template</h3>
				</div>
				<div class="grid grid-cols-2 gap-5 md:grid-cols-3">
					{#each allTemplates as opt, i}
						<button
							class="w-full"
							class:ring={($currentAvaTar.collection ===
								CollectionType.TemplateCustom &&
								$currentVoice.collection === CollectionType.TemplateCustom &&
								$currentAvaTar.id === i &&
								$currentVoice.id === i) ||
								($currentAvaTar.collection === CollectionType.TemplateLibrary &&
									$currentVoice.collection === CollectionType.TemplateLibrary &&
									$currentAvaTar.id === i - $TalkingTemplateCustom.length &&
									$currentVoice.id === i - $TalkingTemplateCustom.length)}
						>
							<BotTemplateCard {...opt} on:click={() => {
								let label =
									i >= $TalkingTemplateCustom.length
										? CollectionType.TemplateLibrary
										: CollectionType.TemplateCustom;
								let no =
									i >= $TalkingTemplateCustom.length
										? i - $TalkingTemplateCustom.length
										: i;
								currentAvaTar.set({ collection: label, id: no });
								currentVoice.set({ collection: label, id: no });
								goto("/chat");
							}} />
						</button>
					{/each}
				</div>
			</div>
		{:else if mode === "customize"}
			<div>
				<h3 class="text-xl font-medium text-[#051F61]">Talking Avatar</h3>
				<div class="flex gap-7 overflow-auto px-2 py-4 text-[#0F172A]">
					{#each allPhotos as opt, i}
						<button
							class:ring={($currentAvaTar.collection ===
								CollectionType.Custom &&
								$currentAvaTar.id === i) ||
								($currentAvaTar.collection === CollectionType.Library &&
									$currentAvaTar.id === i - $TalkingPhotoCustom.length)}
							on:click={() => {
								let label =
									i >= $TalkingPhotoCustom.length
										? CollectionType.Library
										: CollectionType.Custom;
								let no =
									i >= $TalkingPhotoCustom.length
										? i - $TalkingPhotoCustom.length
										: i;
								currentAvaTar.set({ collection: label, id: no });
							}}
						>
							<TalkingPhotoCard {...opt} />
						</button>
					{/each}
				</div>
			</div>
			<div>
				<h3 class="text-xl font-medium text-[#051F61]">Talking Voice</h3>
				<div class="flex gap-7 px-2 py-4 text-[#0F172A]">
					{#each allVoices as opt, i}
						<button
							class:ring={($currentVoice.collection === CollectionType.Custom &&
								$currentVoice.id === i) ||
								($currentVoice.collection === CollectionType.Library &&
									$currentVoice.id === i - $TalkingVoiceCustom.length)}
							on:click={() => {
								let label =
									i >= $TalkingVoiceCustom.length
										? CollectionType.Library
										: CollectionType.Custom;
								let no =
									i >= $TalkingVoiceCustom.length
										? i - $TalkingVoiceCustom.length
										: i;
								currentVoice.set({ collection: label, id: no });
							}}
						>
							<TalkingVoiceCard {...opt} />
						</button>
					{/each}
				</div>
			</div>
			<div>
				<h3 class="text-xl font-medium text-[#051F61]">Knowledge Base</h3>
				<div class="flex gap-7 px-2 py-4 text-[#0F172A]">
					{#each allKnowledges as opt, i}
						<button
							class:ring={($currentKnowledge.collection ===
								CollectionType.Custom &&
								$currentKnowledge.id === i) ||
								($currentKnowledge.collection === CollectionType.Library &&
									$currentKnowledge.id === i - $TalkingKnowledgeCustom.length)}
							on:click={() => {
								let label =
									i >= $TalkingKnowledgeCustom.length
										? CollectionType.Library
										: CollectionType.Custom;
								let no =
									i >= $TalkingKnowledgeCustom.length
										? i - $TalkingKnowledgeCustom.length
										: i;
								currentKnowledge.set({ collection: label, id: no });
							}}
						>
							<TalkingKnowledgeCard {...opt} />
						</button>
					{/each}
				</div>
			</div>
			<button
				class="btn -mt-20 inline-flex w-46 items-center self-end rounded-lg bg-blue-700 text-center font-medium text-white hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300 "
				on:click={() => goto("/chat")}
				>let's talking! -> </button
			>
		{/if}
	</div>
</div>
