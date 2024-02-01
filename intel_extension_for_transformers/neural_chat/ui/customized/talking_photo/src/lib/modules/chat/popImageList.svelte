<script lang="ts">
	import { Button, Modal, Progressbar } from "flowbite-svelte";
	import { currentMode, imageList, popupModal } from "$lib/shared/stores/common/Store";
	import ChatUploadImages from "./ChatUploadImages.svelte";
	import { createEventDispatcher } from "svelte";
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";
	import ImageIcon from "$lib/assets/chat/svelte/ImageIcon.svelte";
	import { getNotificationsContext } from "svelte-notifications";

	const { addNotification } = getNotificationsContext();

	export let currentDragImageList: boolean[]

	const dispatch = createEventDispatcher();
	let uploadHandle: number;
	let uploadProgress = 0;

	function refreshImages(idx: number, imgSrc: string) {
		$imageList[idx].image_path =
			"https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif";

		setTimeout(function () {
			$imageList[idx].image_path = imgSrc;
		}, 2000);
	}

	function handleUploadBegin() {
		uploadHandle = setInterval(() => {
			if (uploadProgress < 70) uploadProgress += 5;
			else if (uploadProgress < 90) uploadProgress += 2;
			else if (uploadProgress < 99) uploadProgress += 1;
		}, 500);
	}

	function handleUploadEnd() {
		uploadProgress = 0;
		clearInterval(uploadHandle);
		addNotification({
				text: "Uploaded successfully",
				position: "bottom-center",
				type: "success",
				removeAfter: 1000,
			});
		dispatch('refreshPrompt')
	}
	
</script>

<button
	class="image-btn h-full"
	on:click={() => {
		popupModal.set(true);
	}}
>
	<!-- <ImageIcon /> -->
</button>

<Modal title="Photo Album" bind:open={$popupModal} autoclose>
	<div class="text-center">
		<div class="flex h-full w-full flex-row items-start">
			{#if ($imageList.length === 0)  || ($currentMode === "Search")}
			<ChatUploadImages
				on:uploadBegin={handleUploadBegin}
				on:uploadEnd={handleUploadEnd}
			/>
			{/if}
			{#each $imageList as image, idx}
				<div class="block shrink-0">
					<!-- svelte-ignore a11y-click-events-have-key-events -->
					<div class="relative"   on:click={() => dispatch("clickImage", idx)}>
						<input
							type="checkbox"
							checked={currentDragImageList[idx]}
							on:click={e => e.stopPropagation()}	
							on:change={() => dispatch("clickImage", idx)}
							class="form-checkbox absolute left-2 top-2 z-50 h-3 w-3 rounded-full"
						/>

						<button on:click={() => dispatch("clickVideoImage", idx)}>
							<img
								alt=""
								class="aspect-square h-[6vw] w-[6vw] object-cover p-2 max-sm:h-[28vw] max-sm:w-[28vw]"
								src={image.image_path}
								data-id={image.image_id}
								on:error={() => {
									refreshImages(idx, image.image_path);
								}}
							/>
						</button>
						<div class="itmes-center absolute inset-0 flex justify-center">
							<span />
						</div>
						<PictureEnlarge
							imgSrc={image.image_path}
							enlargeClass={"w-3 h-3"}
							extraClass={"left-3 bottom-5"}
						/>
					</div>
				</div>
			{/each}
		</div>
	</div>
	{#if uploadProgress}
		<Progressbar
			progress={uploadProgress.toString()}
			size="h-1"
			color="blue"
			class="mb-2"
		/>
	{/if}
	<svelte:fragment slot="footer">
		<Button>Confirm</Button>
		<Button color="alternative">Cancel</Button>
	</svelte:fragment>
</Modal>
