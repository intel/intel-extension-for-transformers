<script lang="ts">
	import { Icon } from "flowbite-svelte-icons";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import Draggable from "$lib/shared/components/drag-drop/Draggable.svelte";

	import UploadImages from "$lib/modules/chat/ChatUploadImages.svelte";
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";

	import { hintUploadImg, imageList } from "$lib/shared/stores/common/Store";
	import { createEventDispatcher } from "svelte";
	import HintBubble from "$lib/shared/components/hint/HintBubble.svelte";

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
	class={`${extraClass} flex h-full flex-col border-l p-2 py-0 max-sm:border-none sm:w-1/3 sm:p-6 w-full`}
>
	<div class="relative my-2 mb-3 flex items-center justify-between">
		<div class="mb-2 text-sm font-semibold leading-none text-gray-400">
			Upload / Drag an image
		</div>
		<button
			on:click={() => dispatch("clickSend")}
			class="text-primary-600 dark:text-primary-500 dark:hover:text-primary-600 hover:text-primary-700 flex items-center text-sm font-medium"
		>
			Send
		
		</button>
	</div>

	<Scrollbar
		className="h-44 sm:grow"
		classLayout="grid grid-cols-4 sm:grid-cols-3 gap-3"
	>
		<UploadImages on:uploadBegin on:uploadEnd />
		{#each $imageList as image, idx}
			<div class="relative">
				<input
					type="checkbox"
					on:change={() => dispatch("clickImage", idx)}
					class="form-checkbox absolute left-0 top-0 h-3 w-3 rounded-full"
				/>

				<Draggable>
					<img
						alt=""
						class="aspect-square object-cover"
						src={image.image_path}
						data-id={image.image_id}
						on:error={() => {
							refreshImages(idx, image.image_path);
						}}
					/>
				</Draggable>

				<PictureEnlarge
					imgSrc={image.image_path}
					enlargeClass={"w-3 h-3"}
					extraClass={"right-0 top-0"}
				/>
			</div>
		{/each}
	</Scrollbar>
</div>
