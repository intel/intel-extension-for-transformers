<script lang="ts">
	import { DOMAIN_LIST } from "$lib/shared/shared.constant";
	
	export let domain_idx: number;
</script>

<div class="grid gap-6 grid-cols-1 text-title px-2 mt-4">
	{#each DOMAIN_LIST as domain, idx}
		<button
			class="w-full text-gray-300 hover:bg-gray-500 flex items-center px-2 py-3 text-sm font-medium rounded-md mb-2 shadow-sm ring-1 ring-inset ring-gray-400 text-white"
			class:bg-[#224c75]={idx === domain_idx}
			class:btn-disabled={idx > 1}
			on:click={() => (domain_idx = idx)}
		>
			<img src={domain.svg} class="w-5 mx-2" alt="" />
			{domain.title}
		</button>
	{/each}
</div>
