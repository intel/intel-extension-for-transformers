<script lang="ts">
	import Knowledge from "./imgs/Knowledge.svelte";
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import EditIcon from "./imgs/Edit.svelte";
	import DeleteIcon from "./imgs/Delete.svelte";

	export let name: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	const dispatch = createEventDispatcher();
    let inputEl: HTMLInputElement

	onMount(() => {
		if (needChangeName) {
			changeName();
		}
	});
    afterUpdate(() => {
		if (inputEl) {
			inputEl.focus();
			inputEl.onblur = () => {
				needChangeName = false;
			};
		}
	});

	function changeName() {
		needChangeName = true;
	}
</script>

<div
	class="relative flex h-32 w-32 flex-col items-center rounded-xl pt-6 shadow-[0_2px_30px_0_rgba(0,0,0,0.1)]"
>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<div
		class="flex items-center justify-center"
	>
        <Knowledge />
	</div>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	{#if notLibrary}
		<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
	{/if}
	<!-- {#if needChangeName}
		<input
			type="text"
			bind:value={name}
			bind:this={inputEl}
			class="mt-2 w-full text-center text-sm text-gray-600 focus-visible:outline-[#ccc] dark:text-gray-400 "
		/>
	{:else} -->
		<!-- <span
			class="relative mt-4 w-9/12 text-sm text-gray-600 dark:text-gray-400  text-ellipsis overflow-hidden whitespace-nowrap"
			on:dblclick|capture={changeName}
		>
			{name}
			{#if notLibrary}
				<span class="absolute -right-2 -top-1"><EditIcon on:changeName={changeName} /></span>
			{/if}
		</span>
	{/if} -->
	<span
			class="relative mt-4 w-9/12 text-sm text-gray-600 dark:text-gray-400  text-ellipsis overflow-hidden whitespace-nowrap"
		>
			{name}
	</span>
</div>