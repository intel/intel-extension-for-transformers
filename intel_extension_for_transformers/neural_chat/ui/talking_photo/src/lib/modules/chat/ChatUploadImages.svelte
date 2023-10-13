<script lang="ts">
	import Add from "$lib/assets/image-info/svelte/Add.svelte";
	import { handleImageUpload } from "$lib/network/image/UploadImage";
	import UploadImageBlobs from "$lib/shared/components/upload/UploadImageBlobs.svelte";
	import {
		hintUploadImg,
		imageList,
		isLoading,
		newUploadNum,
	} from "$lib/shared/stores/common/Store";
	import HintBubble from "$lib/shared/components/hint/HintBubble.svelte";
	import { createEventDispatcher, onMount } from "svelte";
	import { getNotificationsContext } from "svelte-notifications";

	const { addNotification } = getNotificationsContext();
	const dispatch = createEventDispatcher();
	// let progress = 0.0;
	function handleUploadClick(e) {
		hintUploadImg.set(false);
		isLoading.set(true);

		new Promise((resolve) => {
			handleImageUpload(e, resolve);
		}).then(() => {
			isLoading.set(false);
			addNotification({
				text: "Uploaded successfully",
				position: "bottom-center",
				type: "success",
				removeAfter: 1000,
			});
			dispatch("uploadEnd");
		});
		newUploadNum.set(1);
		dispatch("uploadBegin");
	}

</script>

<!-- {#if $hintUploadImg}
<div
    class="absolute flex h-10 w-10 animate-bounce items-center justify-center rounded-full bg-white p-2 shadow-lg ring-1 ring-slate-900/5"
>
    <svg
        class="h-6 w-6 text-violet-500"
        fill="none"
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="2"
        viewBox="0 0 24 24"
        stroke="currentColor"
    >
        <path d="M19 14l-7 7m0 0l-7-7m7 7V3" />
    </svg>
</div>
{/if} -->
<!-- {window.deviceType === 'pc' ? 'image-btn' : ''} -->
<div class="relative">
	<div
		class="relative top-0 flex h-full w-full flex-col items-center justify-center bg-gray-300 p-4 opacity-95 sm:p-6"
	>
		<div class="absolute h-full w-full opacity-0">
			<UploadImageBlobs on:upload={handleUploadClick} />
		</div>
		<Add extraClass="h-7 w-7" />
		<p class="text-xs font-bold text-gray-500">Upload</p>
	</div>
</div>
