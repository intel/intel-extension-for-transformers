<script lang="ts">
	import {
		currentTemplate,
		TemplateCustom,
		CollectionType,
	} from "$lib/shared/stores/common/Store";
	import { TalkingTemplateLibrary } from "$lib/shared/constant/Data";
	import BotTemplateCard from "$lib/modules/side-page/BotTemplateCard.svelte";
	import CreateTemplate from "$lib/modules/side-page/CreateTemplate.svelte";
	import TemplateIcon from "$lib/assets/customize/svelte/TemplateIcon.svelte";

	const customNum = $TemplateCustom.length;

	function handleTemplateDelete(i: number) {
		console.log('i', i);
		console.log('TemplateCustom', $TemplateCustom);
		
		TemplateCustom.update((options) => {
			options.splice(i, 1);
			return options;
		});
	}
</script>

<div class="h-full w-full p-5 sm:mx-5 overflow-auto">
	<div class="mb-1 flex w-full flex-row flex-wrap justify-between sm:mb-0">
		<h2 class="mb-8 text-[1.3rem] font-medium leading-tight text-[#051F61]">
			My ChatBot
		</h2>
	</div>
	{#if $TemplateCustom.length === 0}
		<div class="flex gap-5 p-2">
			<TemplateIcon extraClass="h-32" />
			<div class="flex flex-col items-start justify-center gap-5">
				<h3 class="text-xl">Customize chatbot!</h3>
				<CreateTemplate>
					<div
						class="flex cursor-pointer items-center gap-2 rounded px-2 text-[#8c75ff] ring-1 ring-[#8c75ff]"
					>
						<svg
							class="mx-auto h-5 w-5"
							viewBox="0 0 1024 1024"
							version="1.1"
							xmlns="http://www.w3.org/2000/svg"
							fill="currentColor"
							><path
								d="M1022.955204 556.24776c0 100.19191-81.516572 181.698249-181.718715 181.698249l-185.637977 0c-11.2973 0-20.466124-9.168824-20.466124-20.466124 0-11.307533 9.168824-20.466124 20.466124-20.466124l185.637977 0c77.628008 0 140.786467-63.148226 140.786467-140.766001 0-77.423347-62.841234-140.448776-140.203182-140.766001-0.419556 0.030699-0.828878 0.051165-1.248434 0.061398-5.935176 0.153496-11.665691-2.302439-15.666818-6.702656-4.001127-4.41045-5.884011-10.345626-5.157463-16.250102 1.330298-10.806113 1.944282-19.760043 1.944282-28.192086 0-60.763922-23.658839-117.874641-66.617234-160.833035-42.968627-42.958394-100.089579-66.617234-160.843268-66.617234-47.368844 0-92.742241 14.449084-131.208321 41.781592-37.616736 26.738991-65.952084 63.700811-81.925894 106.884332-2.425236 6.54916-8.012488 11.399631-14.827707 12.893658-6.815219 1.483794-13.927197-0.603751-18.859533-5.536087-19.289322-19.340487-44.943608-29.982872-72.245418-29.982872-56.322773 0-102.146425 45.813419-102.146425 102.125959 0 0.317225 0.040932 0.982374 0.092098 1.627057 0.061398 0.920976 0.122797 1.831718 0.153496 2.762927 0.337691 9.465582-5.863545 17.928325-15.001669 20.455891-32.356942 8.943696-61.541635 28.550243-82.181721 55.217602-21.305235 27.516704-32.571836 60.508096-32.571836 95.41307 0 86.244246 70.188572 156.422585 156.443052 156.422585l169.981393 0c11.2973 0 20.466124 9.15859 20.466124 20.466124 0 11.2973-9.168824 20.466124-20.466124 20.466124l-169.981393 0c-108.828614 0-197.3753-88.536452-197.3753-197.354833 0-44.053332 14.223956-85.712127 41.126676-120.473839 22.809495-29.450752 53.897537-52.086285 88.710414-64.816215 5.065366-74.322729 67.149353-133.2447 142.751215-133.2447 28.386514 0 55.504128 8.217149 78.651314 23.52581 19.657712-39.868009 48.842405-74.169233 85.497233-100.212376 45.434795-32.295544 99.004875-49.354058 154.918325-49.354058 71.692832 0 139.087778 27.915793 189.782368 78.600149 50.694589 50.694589 78.610382 118.089535 78.610382 189.782368 0 3.704368-0.102331 7.470135-0.296759 11.368932C952.633602 386.245901 1022.955204 463.188294 1022.955204 556.24776z"
							/><path
								d="M629.258611 589.106122c-3.990894 3.990894-9.230222 5.996574-14.46955 5.996574s-10.478655-2.00568-14.46955-5.996574l-67.087954-67.077721 0 358.689289c0 11.307533-9.15859 20.466124-20.466124 20.466124-11.307533 0-20.466124-9.15859-20.466124-20.466124l0-358.689289-67.087954 67.077721c-7.992021 7.992021-20.947078 7.992021-28.939099 0s-7.992021-20.957311 0-28.949332l102.023628-102.013395c7.992021-7.992021 20.947078-7.992021 28.939099 0l102.023628 102.013395C637.250632 568.148811 637.250632 581.114101 629.258611 589.106122z"
							/></svg
						>
						Create chatbot
					</div>
				</CreateTemplate>
			</div>
		</div>
	{:else}
		<div class="grid grid-cols-2 gap-4">
			<CreateTemplate />
			{#each $TemplateCustom as opt, i (opt)}
				<div class="block shrink-0">
					<button
						class="aspect-video w-full rounded-2xl"
						class:ring={$currentTemplate.collection ===
							CollectionType.Custom && $currentTemplate.id === i}
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
					</button>
				</div>
			{/each}
		</div>
	{/if}

	<div
		class="mb-1 mt-6 flex w-full flex-row flex-wrap justify-between sm:mb-0"
	>
		<h2 class="mb-8 text-[1.3rem] font-medium leading-tight text-[#051F61]">
			ChatBot Library
		</h2>
	</div>
	<div class="grid grid-cols-2 gap-4 sm:grid-cols-3">
		{#each TalkingTemplateLibrary as opt, i}
			<div
				class="aspect-video h-full w-full rounded-2xl"
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
</div>
