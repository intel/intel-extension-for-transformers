<script lang="ts">
	import { TalkingKnowledgeLibrary } from "$lib/components/talkbot/constant";
	import TalkingKnowledgeCard from "$lib/components/talkbot/talking-knowledge-card.svelte";
	import { CollectionType, currentKnowledge, TalkingKnowledgeCustom } from "$lib/components/talkbot/store";
	import knowledgeIcon from "$lib/assets/knowledge.svg";
	import UploadKnowledge from "$lib/components/talkbot/upload-knowledge.svelte";
    import { fetchKnowledgeBaseId } from "$lib/modules/chat/network";

	const customNum = $TalkingKnowledgeCustom.length
    let loading = false

    async function handleKnowledgeUpload(e: CustomEvent<any>) {
        loading = true
		let knowledge_id = "";
		try {
			const blob = await fetch(e.detail.src).then(r => r.blob());
        	const res = await fetchKnowledgeBaseId(blob);
			knowledge_id = res.knowledge_id ? res.knowledge_id : "default";

		} catch {
			knowledge_id = "default";
		}

        loading = false

		TalkingKnowledgeCustom.update((options) => {
			return [{ name: e.detail.fileName, src: e.detail.src, id: knowledge_id }, ...options];
		});
	}

    function handleKnowledgeDelete(i :number) {
		TalkingKnowledgeCustom.update(options => {
			options.splice(i, 1)
			return options;
		});
	}
</script>

<div class="flex h-full">
	<div class="h-full w-full p-5 shadow sm:mx-5 xl:mx-20">
		<div class="mb-1 flex w-full flex-row flex-wrap justify-between sm:mb-0">
			<h2
				class="mb-6 text-2xl font-medium leading-tight text-[#051F61] md:pr-0"
			>
				My Knowledge Base
			</h2>
		</div>
		{#if $TalkingKnowledgeCustom.length === 0}
			<div class="flex gap-10 p-2">
				<img class="h-32" src={knowledgeIcon} alt="" />
				<div class="flex flex-col items-start justify-center gap-5">
					<h3 class="text-2xl">Upload your own Knowledge Base!</h3>
					<UploadKnowledge on:upload={handleKnowledgeUpload}>
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
                            Upload Knowledge Base
                        </div>
                    </UploadKnowledge>
                    <span class="loading loading-bars loading-md text-[#8c75ff]" class:hidden={!loading}></span>
				</div>
			</div>
		{:else}
			<div class="flex gap-7 py-4 text-[#0F172A]">
				<UploadKnowledge on:upload={handleKnowledgeUpload} />
                <span class="loading loading-bars loading-md text-[#8c75ff]" class:hidden={!loading}></span>
				<div class="flex gap-7">
					{#each $TalkingKnowledgeCustom as opt, i (opt.name + i)}
						<button
							class:ring={$currentKnowledge.collection === CollectionType.Custom &&
								$currentKnowledge.id === i}
							on:click={() => {
								currentKnowledge.set({ collection: CollectionType.Custom, id: i });
							}}
						>
							<TalkingKnowledgeCard
                                bind:name={opt.name}
                                needChangeName={i >= customNum}
                                notLibrary
								on:delete={() => handleKnowledgeDelete(i)}
							/>
						</button>
					{/each}
				</div>
			</div>
		{/if}

		<div
			class="mb-1 mt-12 flex w-full flex-row flex-wrap justify-between sm:mb-0"
		>
			<h2
				class="mb-6 text-2xl font-medium leading-tight text-[#051F61] md:pr-0"
			>
				Knowledge Base Library
			</h2>
			<!-- <div class="text-end">
                <form class="flex w-full space-x-3">
                    <div class=" relative ">
                        <input
                            type="text"
                            id="form-subscribe-Filter"
                            class=" rounded-lg border-transparent flex-1 appearance-none border border-gray-300 w-full py-2 px-4 bg-white placeholder-gray-400 shadow-sm text-base focus:outline-none focus:ring-1 focus:ring-purple-600 focus:border-transparent"
                            placeholder="name"
                        />
                    </div>
                    <button
                        class="flex-shrink-0 px-4 py-2 text-base font-semibold text-white bg-gray-600 rounded-lg shadow-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-purple-200"
                        type="submit"
                    >
                        Filter
                    </button>
                </form>
            </div> -->
		</div>
		<div class="flex gap-7 overflow-auto px-2 py-4 text-[#0F172A]">
			{#each TalkingKnowledgeLibrary as opt, i (opt.name + i)}
				<button
					class:ring={$currentKnowledge.collection === CollectionType.Library &&
						$currentKnowledge.id === i}
					on:click={() => {
						currentKnowledge.set({ collection: CollectionType.Library, id: i });
					}}
				>
					<TalkingKnowledgeCard
						{...opt}
					/>
				</button>
			{/each}
		</div>
	</div>
</div>
