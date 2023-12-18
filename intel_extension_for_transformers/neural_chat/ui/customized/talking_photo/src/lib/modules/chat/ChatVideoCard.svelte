<script lang="ts">
	import { Icon } from "flowbite-svelte-icons";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import Draggable from "$lib/shared/components/drag-drop/Draggable.svelte";

	import UploadImages from "$lib/modules/chat/ChatUploadImages.svelte";
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";

	import { hintUploadImg, imageList } from "$lib/shared/stores/common/Store";
	import { createEventDispatcher } from "svelte";
	import HintBubble from "$lib/shared/components/hint/HintBubble.svelte";
	import Warning from "$lib/assets/chat/svelte/Warning.svelte";
	import Close from "$lib/assets/chat/svelte/Close.svelte";


	const dispatch = createEventDispatcher();

	export let extraClass = "";

	function refreshImages(idx: number, imgSrc: string) {
		$imageList[idx].image_path =
			"https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif";

		setTimeout(function () {
			$imageList[idx].image_path = imgSrc;
		}, 2000);
	}
</script>

<div
	class={`${extraClass} flex h-full w-full flex-col p-2 py-0 border-none relative`}
>
	<button
		class="absolute right-0 top-0 z-50 rounded-full bg-[#eeeeeec7] p-1"
		on:click={() => dispatch("closeTool")}><Close /></button
	>
	{#if $imageList.length === 0}
		<div class="flex flex-col items-center py-4">
			<Warning extraClass="w-36 h-36" />
			<p class="text-xs text-gray-500">
				Oops! You haven't uploaded any images yet.
			</p>
			<p class="text-sm text-gray-500">Please upload one first.</p>
		</div>
	{:else}
		<div class="relative my-2 mb-3 flex items-center justify-between">
			<div class="mb-2 text-sm font-semibold leading-none text-gray-400">
				Select an Image
			</div>
		</div>
		<Scrollbar
			className="h-44 sm:grow"
			classLayout="grid grid-cols-4 sm:grid-cols-3 gap-3"
		>
			{#each $imageList as image, idx}
				<div class="relative">
					<button on:click={() => dispatch("clickVideoImage", idx)}>
						<img
							alt=""
							class="aspect-square object-cover"
							src={image.image_path}
							data-id={image.image_id}
							on:error={() => {
								refreshImages(idx, image.image_path);
							}}
						/>
					</button>

					<PictureEnlarge
						imgSrc={image.image_path}
						enlargeClass={"w-3 h-3"}
						extraClass={"right-0 top-0"}
					/>
				</div>
			{/each}
		</Scrollbar>
	{/if}
</div>
