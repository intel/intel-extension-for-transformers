<script lang="ts">
	import ScrollOptions from "$lib/shared/components/scroll-options.svelte";

	export let optionType: string;
	export let selected: any;
	export let article: string;
	export let MODEL_OPTION: {
		names: string[];
		options: {
			label: string;
			value: number;
			minRange: number;
			maxRange: number;
			step: number;
			type: string;
		}[];
	};
	export let KNOWLEDGE_OPTIONS: string[];

	function handleFileSubmit(event:Event & { currentTarget: EventTarget & HTMLInputElement}) {
		if ((event.target as HTMLInputElement).files?.length == 0) return;
		let reader = new FileReader();
		reader.onload = (e) => {
			article = e.target?.result as string;
		}
		reader.readAsText(((event.target as HTMLInputElement).files as FileList)[0]);
	}

</script>

<div class="flex flex-col min-h-0 grow gap-5">
	<ScrollOptions
		lists={MODEL_OPTION}
		bind:selected={selected["Model"]}
	/>
	<div>
		<div class="w-full flex items-center gap-2">
			<input type="radio" class="radio radio-primary"
				checked={optionType === 'knowledge base'}
				on:click={() => {optionType = optionType === 'Model' ? 'knowledge base' : 'Model'}}
			/>
			{#if optionType === 'knowledge base'}
				<select
					class="w-32 select bg-transparent text-white border-white p-2"
					bind:value={selected['knowledge base']}
				>
					{#each KNOWLEDGE_OPTIONS as name}
						<option class="bg-transparent text-black" value={name}>
							{name}
						</option>
					{/each}
				</select>
			{:else}
				<p class="text-white ml-2 border border-white rounded-lg p-2">Knowledge Base</p>
			{/if}
		</div>
		{#if selected[optionType] === 'Customized'}
			<p class="text-white my-3">Customized: </p>
			<input class="w-full input-range bg-blue-400 border-none" type="file" on:change={handleFileSubmit}/>
		{/if}
	</div>

</div>
