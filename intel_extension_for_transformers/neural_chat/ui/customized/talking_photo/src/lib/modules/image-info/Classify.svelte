<script lang="ts">
	import {
		fetchImagesByType,
		updateLabel,
	} from "$lib/network/image/Network";
	import { getTypeList } from "$lib/network/image/getTypeLists";
	import InlineEditable from "$lib/shared/components/editable/InlineEditable.svelte";
	import { imageList } from "$lib/shared/stores/common/Store";
	import { Input } from "flowbite-svelte";
	export let typeList: { [index: string]: { [index: string]: string } };
	export let shownType: string;
	export let hiddenAlbum: Boolean;
	export let firstTypeName: string = "";

	async function handleTypeClick(type: string, subtype: string) {
		let res = await fetchImagesByType(type, subtype);
		shownType = subtype;
		imageList.set(res);
		hiddenAlbum = true;
	}

	async function handleTypeEdit(label: string, from: string, to: string) {
		if (shownType === from) {
			shownType = to;
		}
		await updateLabel(label, from, to);
		typeList = await getTypeList();
	}
</script>

<div
	class="mr-24 mt-6 grid grid-cols-5 gap-7 max-sm:mr-0 max-sm:grid-cols-2 max-sm:gap-5 max-sm:px-4"
>
	{#each Object.keys(typeList[firstTypeName]) as secondaryType}
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<div
			class="max-w-sm rounded-lg border border-gray-200 bg-white shadow"
			
		>
			<div class="relative overflow-hidden" on:click={() => handleTypeClick(firstTypeName, secondaryType)}>
				<img
					class="h-[7rem] w-full rounded-t-lg object-cover"
					src={typeList[firstTypeName][secondaryType]}
					alt=""
				/>
				<div
					class="absolute bottom-0 left-0 right-0 top-0 h-full w-full max-w-sm overflow-hidden rounded-t-lg bg-black bg-fixed opacity-30"
				/>
			</div>

			<div class="p-2">
				{#if firstTypeName === "time"}
					<Input
						name="date"
						class="w-full border-none bg-transparent p-0 text-base font-medium text-gray-900"
						required
						type="date"
						bind:value={secondaryType}
						on:change={(e) => {
							handleTypeEdit(firstTypeName, secondaryType, e.target.value);
						}}
					/>
				{:else}
					<InlineEditable
						extraClass="text-sm  h-5"
						inputExtraClass="block w-full p-2.5 text-base font-medium text-gray-900"
						bind:value={secondaryType}
						showIcon
						on:change={(e) => {
							handleTypeEdit(firstTypeName, secondaryType, e.target.value);
						}}
					/>
				{/if}
			</div>
		</div>
	{/each}
</div>
