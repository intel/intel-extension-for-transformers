<script lang="ts">
	import {
		currentTemplate,
		TemplateCustom,
		CollectionType,
		showTemplate
	} from "$lib/shared/stores/common/Store";
	import { TalkingTemplateLibrary } from "$lib/shared/constant/Data";
	import BotTemplateCard from "$lib/modules/side-page/BotTemplateCard.svelte";
	import CreateTemplate from "$lib/modules/side-page/CreateTemplate.svelte";
	import TemplateIcon from "$lib/assets/customize/svelte/TemplateIcon.svelte";
	import Template from "$lib/modules/customize/Template.svelte";

	const customNum = $TemplateCustom.length;
	console.log('TemplateCustom', $TemplateCustom);
	console.log('TalkingTemplateLibrary', TalkingTemplateLibrary);
	

	function handleTemplateDelete(i: number) {
		console.log("i", i);
		console.log("TemplateCustom", $TemplateCustom);

		TemplateCustom.update((options) => {
			options.splice(i, 1);
			return options;
		});
	}

</script>

<div
	class="h-full w-full overflow-auto bg-white p-16 pb-0 sm:mx-5 lg:rounded-tl-3xl"
>
	<div
		class={`flex w-[15rem] flex-col gap-4 rounded-2xl border p-5
                    ${$showTemplate ? "border-4 border-[#9fc1fb]" : ""} `}
		style="background: url(&quot;https://cdn.heygen.com/heygen/home/home-template-bg.png&quot;) center center / cover no-repeat;"
		on:click={() => {
			showTemplate.set(!$showTemplate);
		}}
	>
		<img
			class="h-12 w-12"
			src="https://imgur.com/qCPlF9s.png"
			alt=""
		/>
		<div class="">
			<div class="text-left text-lg font-bold">Customize Chatbot</div>
		</div>
	</div>

	{#if $showTemplate}
		<Template />
	{:else}
		<div
			class="mb-1 mt-6 flex w-full flex-row flex-wrap justify-between sm:mb-0"
		>
			<h2 class="mb-8 text-[1.3rem] font-medium leading-tight text-[#051F61]">
				Available ChatBots
			</h2>
		</div>
		<div class="gap-4 flex-wrap flex-wrap flex flex-row">
			{#each $TemplateCustom as opt, i (opt)}
				<div class="block shrink-0">
					<div
						class="rounded-2xl sm:w-[15rem]"
						class:ring={$currentTemplate.collection === CollectionType.Custom &&
							$currentTemplate.id === i}
					>
						<BotTemplateCard
							{...opt}
							index={i}
							bind:name={opt.name}
							needChangeName={i >= customNum}
							notLibrary
							on:delete={() => handleTemplateDelete(i)}
							on:click={() => {
								currentTemplate.set({
									collection: CollectionType.Custom,
									id: i,
								});
							}}
						/>
					</div>
				</div>
			{/each}
			{#each TalkingTemplateLibrary as opt, i}
				<div
					class="aspect-video h-full w-full sm:w-[15rem] rounded-2xl"
					class:ring={$currentTemplate.collection === CollectionType.Library &&
						$currentTemplate.id === i}
				>
					<BotTemplateCard
						{...opt}
						on:click={() => {
							currentTemplate.set({
								collection: CollectionType.Library,
								id: i,
							});
						}}
					/>
				</div>
			{/each}
		</div>
	{/if}
</div>
