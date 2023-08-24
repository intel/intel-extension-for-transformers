<script lang="ts">
	import { DEFAULT_TIP } from "$lib/shared/shared.constant";
	import { createEventDispatcher } from 'svelte';

	export let mode: string;
	const dispatch = createEventDispatcher();
</script>

<div class="flex flex-col-reverse items-start sm:flex-row">
	<div class="flex flex-col pr-8 mb-4">
		<h1
			class="mb-3 text-3xl font-bold tracking-tight text-black dark:text-white md:text-5xl"
		>
			Welcome

			<span class="relative inline-block ml-2">
				<span class="relative text-blue-600 skew-y-3"
					>Neural Chat</span
				>
			</span>
			!
		</h1>
		<p class="mb-16 text-gray-600 dark:text-gray-400">
			<a href="#">Your AI-powered copilot for the web 😊</a>
		</p>
	</div>
	<div
		class="w-[5rem] h-[5rem] rounded-full sm:w-[11rem] sm:h-[8.5rem] relative mb-8 sm:mb-0 mr-auto bg-cyan-300 bg-opacity-25"
	>
		<img src="" alt="" />
	</div>
</div>
{#if mode == "basic" || mode == "advanced"}
	<section class="w-full mb-16">
		<h3
			class="mb-6 text-2xl font-bold tracking-tight text-black dark:text-white md:text-4xl"
		>
			Hints
		</h3>
		<div class="flex flex-col gap-6 md:flex-row">
			{#each DEFAULT_TIP as tip}
				<!-- svelte-ignore a11y-click-events-have-key-events -->
				<div
					class="w-full transform rounded-xl bg-gradient-to-r from-sky-400 via-blue-500 to-purple-500 p-1 transition-all hover:scale-[1.01] md:w-1/3"
					on:click={()=>dispatch('tipclick', tip.name)}
				>
					<div
						class="flex h-full flex-col justify-between rounded-lg bg-white p-4 dark:bg-gray-900"
					>
						<div class="flex flex-col justify-between md:flex-row">
							<h4
								class="mb-6 w-full text-lg font-medium tracking-tight text-gray-900 dark:text-gray-100 sm:mb-10 md:text-lg"
							>
								{tip.name}
							</h4>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</section>
{/if}
