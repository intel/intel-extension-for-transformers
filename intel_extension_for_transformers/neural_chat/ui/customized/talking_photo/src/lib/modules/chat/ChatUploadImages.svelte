<script lang="ts">
	import ColorImg from "$lib/assets/chat/svelte/ColorImg.svelte";
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
	import ImageIcon from "$lib/assets/chat/svelte/ImageIcon.svelte";

	export let extraClass = "";
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
			dispatch("uploadEnd");
		});
		newUploadNum.set(1);
		dispatch("uploadBegin");
	}

</script>

<div class="block shrink-0 border-dashed border-4 border-indigo-500">
	<UploadImageBlobs on:upload={handleUploadClick} />
</div>
