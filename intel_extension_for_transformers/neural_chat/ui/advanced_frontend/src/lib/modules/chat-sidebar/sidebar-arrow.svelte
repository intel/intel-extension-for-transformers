<script lang="ts">
	export let isBoardVisible: boolean;
	export let isHistoryVisible: boolean;
	export let label: string;
</script>

{#if label == "leftArrow"}
	<button
		class="absolute right-0 top-1/2 -translate-y-1/2 h-20 bg-slate-700 rounded-l-3xl py-5 cursor-pointer"
		class:rotate-180={!isBoardVisible}
		class:translate-x-full={!isBoardVisible}
		on:click={() => {
			isBoardVisible = !isBoardVisible;
		}}
	>
		<span class="text-white text-3xl">&#9666;</span>
	</button>
{:else}
	<button
		class="absolute left-0 top-1/2 -translate-y-1/2 h-20 bg-slate-700 rounded-r-3xl py-5 cursor-pointer"
		class:rotate-180={!isHistoryVisible}
		class:-translate-x-full={!isHistoryVisible}
		on:click={() => {
			isHistoryVisible = !isHistoryVisible;
		}}
	>
		<span class="text-white text-3xl">&#9656;</span>
	</button>
{/if}
