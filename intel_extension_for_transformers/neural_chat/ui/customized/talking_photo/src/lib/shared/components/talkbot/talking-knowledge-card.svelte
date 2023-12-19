<script lang="ts">
	import Knowledge from "$lib/assets/knowledge/svelte/Knowledge.svelte";
	import { afterUpdate, createEventDispatcher, onMount } from "svelte";
	import EditIcon from "$lib/assets/avatar/svelte/Edit.svelte";
	import DeleteIcon from "$lib/assets/avatar/svelte/Delete.svelte";

	export let name: string;
	export let notLibrary: boolean = false;
	export let needChangeName = false;

	const dispatch = createEventDispatcher();
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

<div
	class="flex h-full w-full items-center justify-center rounded-xl pt-1 shadow-[0_2px_30px_0_rgba(0,0,0,0.1)]"
>
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<div class="flex w-full flex-col items-center justify-center">
		<Knowledge />
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		{#if notLibrary}
			<DeleteIcon on:DeleteAvatar={() => dispatch("delete")} />
		{/if}
		<span
			class="relative w-full overflow-hidden text-ellipsis whitespace-nowrap text-xs text-gray-600 truncate"
		>
			{name}
		</span>
	</div>
</div>
