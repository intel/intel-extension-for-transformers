<script lang="ts">
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import EditIcon from "./imgs/Edit.svelte";
	import DeleteIcon from "./imgs/Delete.svelte";

	export let name: string;
	export let avatar: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	let dispatch = createEventDispatcher();
	let inputEl: HTMLInputElement;

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

<div class="relative flex w-28 flex-col items-center">
	<div class="relative">
		<span class="relative block">
			<img
				alt={name}
				src={avatar}
				class="mx-auto h-24 w-24 rounded object-cover hover:border"
			/>
			{#if notLibrary}
				<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
			{/if}
		</span>
	</div>
	<!-- {#if needChangeName}
		<input
			type="text"
			bind:value={name}
			bind:this={inputEl}
			class="mt-2 w-full text-center text-xs text-gray-600 focus-visible:outline-[#ccc] dark:text-gray-400"
		/>
	{:else} -->
	<span class="relative mt-2 text-xs text-gray-600 dark:text-gray-400  text-ellipsis overflow-hidden whitespace-nowrap">{name}</span>
		<!-- <span
			class="relative mt-2 text-xs text-gray-600 dark:text-gray-400  text-ellipsis overflow-hidden whitespace-nowrap"
			on:dblclick|capture={changeName}>
			{name}
			{#if notLibrary}
				<span class="absolute -right-3 -top-1"><EditIcon on:changeName={changeName} /></span>
			{/if}
		</span> -->
	<!-- {/if} -->
</div>
