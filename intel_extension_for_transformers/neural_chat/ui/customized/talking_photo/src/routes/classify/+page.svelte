<script lang="ts">
	import {
		imageList,
		newUploadNum,
		type ImgInfo,
		type ImgListPiece,
	} from "$lib/shared/stores/common/Store";
	import { DateInput } from "date-picker-svelte";
	import {
		fetchImageDetail,
		fetchImageList,
		fetchImagesByType,
		updateImageInfo,
		updateLabel,
	} from "$lib/network/image/Network";
	import Charactor from "$lib/assets/image-info/svelte/Charactor.svelte";
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";
	import AreaEditable from "$lib/shared/components/editable/AreaEditable.svelte";
	import InfoTag from "$lib/modules/image-info/InfoTag.svelte";
	import Delete from "$lib/assets/image-info/svelte/delete.svelte";
	import Map from "$lib/assets/image-info/svelte/Map.svelte";
	import { afterUpdate, onMount } from "svelte";
	import {
		checkProcessingImage,
		getTypeList,
	} from "$lib/network/image/getTypeLists";
	import InlineEditable from "$lib/shared/components/editable/InlineEditable.svelte";
	import HintUploadImages from "$lib/modules/image-info/HintUploadImages.svelte";
	import { getImageDetail } from "$lib/network/image/getImageDetail";
	import {
		Input,
		Drawer,
		Button,
		Card,
		ImagePlaceholder,
		Alert,
		Modal,
	} from "flowbite-svelte";
	import { sineIn } from "svelte/easing";
	import { getNotificationsContext } from "svelte-notifications";

	import Classify from "$lib/modules/image-info/Classify.svelte";
	import Category from "$lib/assets/image-info/svelte/Category.svelte";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import { Icon } from "flowbite-svelte-icons";
	const { addNotification } = getNotificationsContext();
	let modelEl: HTMLDialogElement;
	let isModifiedCaption = false;
	let isModifiedTags = false;
	let newTagType: string;
	let newTagValue: string;
	let shownType = "My";
	let done: boolean = false;
	let imgInfo: ImgInfo = {
		image_id: 0,
		image_path: "",
		caption: "",
		checked: true,
		location: "",
		time: "",
		tag_list: [],
	};
	let typeList: { [index: string]: { [index: string]: string } } = {};
	let firstTypeName: string = "ALL";
	let hiddenAlbum = true;
	let makeSureDelete = false;
	let otherImageList: any;

	newUploadNum.set(0);
	let popupModal = false;
	let deleteImg: ImgListPiece;

	let transitionParams = {
		x: -320,
		duration: 200,
		easing: sineIn,
	};

	$: otherImageList = typeList.other;

	function intervalFunction() {
		let intervalHandle = setInterval(async () => {
			[done, typeList] = await checkProcessingImage();
			let res = await fetchImageList();
			if (res) imageList.set(res);

			if (done) {
				clearInterval(intervalHandle);
			}
		}, 500);
	}

	onMount(async () => {
		[done, typeList] = await checkProcessingImage();
		if (!done) {
			setTimeout(intervalFunction, 500);
		}

		let res = await fetchImageList();
		if (res) imageList.set(res);
	});

	function handleEditTag(e: CustomEvent) {
		isModifiedTags = true;
		const changeInfo = e.detail;

		if (imgInfo.tag_list[changeInfo.idx]) {
			imgInfo.tag_list[changeInfo.idx] = [
				changeInfo.newKey,
				changeInfo.newValue,
			];
		}
	}

	function handleAddTag() {
		if (newTagType && newTagValue) {
			isModifiedTags = true;
			imgInfo.tag_list = [[newTagType, newTagValue], ...imgInfo.tag_list];

			newTagType = "";
			newTagValue = "";
		} else {
			addNotification({
				text: "Input can't be empty",
				position: "bottom-center",
				type: "error",
				removeAfter: 1000,
			});
		}
	}

	function refreshImages(idx: number, imgSrc: string) {
		$imageList[idx].image_path =
			"https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif";

		setTimeout(function () {
			$imageList[idx].image_path = imgSrc;
		}, 2000);
	}

	async function selectImage(image_id: string) {
		imgInfo = await getImageDetail(image_id);
		modelEl.showModal();
	}

	async function deleteImage(imageToDelete: ImgListPiece) {
		imageList.update((currentImageList) => {
			return currentImageList.filter(
				(image) => image.image_id !== imageToDelete.image_id
			);
		});
		await updateImageInfo(imageToDelete.image_id, undefined, "/deleteImage");
		addNotification({
			text: "Deleted successfully",
			position: "bottom-center",
			type: "success",
			removeAfter: 1000,
		});
		typeList = await getTypeList();
	}

	function handleDeleteTag(idx: number) {
		isModifiedTags = true;
		imgInfo.tag_list.splice(idx, 1);
		imgInfo.tag_list = imgInfo.tag_list;
	}

	async function handleModelClose() {
		if (isModifiedCaption) {
			await updateImageInfo(
				imgInfo.image_id,
				{ caption: imgInfo.caption },
				"/updateCaption"
			);
		}
		if (isModifiedTags) {
			let tag_list = [
				...imgInfo.tag_list,
				["location", imgInfo.location],
				["time", imgInfo.time],
			];
			let tags = Object.fromEntries(tag_list);
			await updateImageInfo(imgInfo.image_id, { tags }, "/updateTags");
		}
		modelEl.close();
		if (isModifiedCaption || isModifiedTags) {
			isModifiedCaption = isModifiedTags = false;
			typeList = await getTypeList();
		}
	}

	async function getMyImageList() {
		let res = await fetchImageList();
		shownType = "My";
		if (res) imageList.set(res);

		hiddenAlbum = true;
	}
</script>

{#if $imageList.length === 0}
	<HintUploadImages />
{:else}
	<Scrollbar classLayout="" className="h-full mt-5 max-auto lg:pl-24">
		<div class="col-span-3 gap-7 max-sm:col-span-4 max-sm:h-full max-sm:pt-0">
			<div
				class="flex items-center justify-start overflow-auto bg-white p-6 py-2 pb-4"
			>
				<button
					type="button"
					class="mr-4 rounded-full border border-[#3369FF] px-6 py-2 text-center text-sm font-medium {firstTypeName ===
					'ALL'
						? 'bg-[#3369FF] text-white'
						: 'bg-white text-[#3369FF] '}"
					on:click={() => {
						getMyImageList();
						firstTypeName = "ALL";
					}}>ALL</button
				>
				{#each Object.keys(typeList).filter((value) => Object.keys(typeList[value]).length !== 0) as firstType}
					<button
						type="button"
						class="mr-4 rounded-full border border-[#3369FF] px-6 py-2 text-sm font-medium uppercase {firstTypeName ===
						firstType
							? 'bg-[#3369FF] text-white'
							: 'bg-white text-[#3369FF] '}"
						on:click={() => {
							hiddenAlbum = false;
							firstTypeName = firstType;
						}}>{firstType}</button
					>
				{/each}
			</div>

			{#if firstTypeName === "ALL" || hiddenAlbum}
				<div
					class=" grid grid-cols-6 gap-5 p-6 pb-4 pr-24 pt-6 max-sm:grid-cols-3 max-sm:gap-4 max-sm:pr-4"
				>
					{#each $imageList as image, idx}
						<figure class="relative h-full w-full">
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<!-- svelte-ignore a11y-img-redundant-alt -->
							<img
								src={image.image_path}
								class="h-full w-full cursor-pointer rounded-md object-cover hover:border max-sm:h-[6rem] max-sm:w-[6rem]"
								alt=""
								on:click={() => {
									selectImage(image.image_id);
								}}
								on:error={() => {
									refreshImages(idx, image.image_path);
								}}
							/>
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<button
								class="absolute right-0 top-0 rounded-full bg-gray-500 text-white hover:bg-gray-700 active:border"
								on:click={(event) => {
									if (image.image_id !== "") {
										popupModal = true;
										event.stopPropagation();
										deleteImg = image;
									}
								}}
							>
								<Delete />
							</button>
						</figure>
					{/each}
				</div>
			{:else if firstTypeName === "other"}
				<div
					class=" grid grid-cols-6 gap-5 p-6 pb-4 pr-24 pt-6 max-sm:grid-cols-3 max-sm:gap-4 max-sm:pr-4"
				>
					{#each otherImageList as image, idx}
						<figure class="relative h-full w-full">
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<!-- svelte-ignore a11y-img-redundant-alt -->
							<img
								src={image.image_path}
								class="h-full w-full cursor-pointer rounded-md object-cover hover:border max-sm:h-[6rem] max-sm:w-[6rem]"
								alt=""
								on:click={() => {
									selectImage(image.image_id);
								}}
								on:error={() => {
									refreshImages(idx, image.image_path);
								}}
							/>
							<!-- svelte-ignore a11y-click-events-have-key-events -->
							<button
								class="absolute right-0 top-0 rounded-full bg-gray-500 text-white hover:bg-gray-700 active:border"
								on:click={(event) => {
									if (image.image_id !== "") {
										popupModal = true;
										event.stopPropagation();
										deleteImg = image;
									}
								}}
							>
								<Delete />
							</button>
						</figure>
					{/each}
				</div>
			{:else}
				<div class="showModalAlbum sm:row-start-1">
					<Classify
						bind:typeList
						bind:firstTypeName
						bind:shownType
						bind:hiddenAlbum
					/>
				</div>

				<div class="showClassifyAlbum">
					<Classify
						bind:typeList
						bind:shownType
						bind:firstTypeName
						bind:hiddenAlbum
					/>
				</div>
			{/if}
		</div>
	</Scrollbar>
{/if}

<dialog bind:this={modelEl} class="rounded-lg">
	<button
		class="absolute right-0 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-[rgba(0,0,0,.4)] text-lg text-white"
		on:click={handleModelClose}>X</button
	>

	<Card class="relative m-0 p-0 sm:p-0">
		<PictureEnlarge
			imgSrc={imgInfo.image_path}
			extraClass="left-0 max-sm:block"
			enlargeClass={"w-6 h-6"}
		/>

		<img
			src={imgInfo.image_path}
			alt=""
			class="mr-2 h-auto object-cover max-sm:m-0 max-sm:h-[8rem] max-sm:w-auto max-sm:rounded-t-md sm:h-[12rem] sm:w-full"
		/>
		<div class="m-6 flex flex-col max-sm:mt-2">
			<Input
				name="date"
				required
				type="date"
				class="max-w-[8rem] border-none bg-transparent"
				bind:value={imgInfo.time}
				on:change={() => (isModifiedTags = true)}
			/>
			<label
				for="website-admin"
				class="mb-1 mt-2 block text-sm font-medium text-gray-900 dark:text-white"
				>Location</label
			>
			<div class="flex">
				<span
					class="inline-flex items-center rounded-l-md border border-r-0 border-gray-300 px-3 text-sm text-gray-900"
				>
					<Map />
				</span>
				<input
					type="text"
					class="block w-full min-w-0 flex-1 rounded-none rounded-r-lg border border-gray-300 bg-gray-50 p-2.5 text-sm text-gray-900 focus:border-blue-500 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500"
					placeholder="location"
					bind:value={imgInfo.location}
					on:change={() => (isModifiedTags = true)}
				/>
			</div>
			<label
				for="website-admin"
				class="mb-1 mt-2 block text-sm font-medium text-gray-900 dark:text-white"
				>Description</label
			>
			<AreaEditable
				areaExtraClass="font-normal text-gray-500 leading-normal rounded-lg text-sm text-left"
				bind:value={imgInfo.caption}
				on:change={() => (isModifiedCaption = true)}
			/>
			<label
				for="website-admin"
				class="mb-1 mt-2 block text-sm font-medium text-gray-900 dark:text-white"
				>Photo Annotation</label
			>
			<div class="flex items-center gap-2">
				<div class="flex flex-1 justify-between">
					<input
						type="text"
						class="form-input h-full w-1/2 truncate border-0 border-b-[1px] border-[#ddd] bg-transparent py-2 pl-0 pr-4 text-sm focus-visible:ring-0"
						placeholder="Type"
						bind:value={newTagType}
					/>

					<input
						type="text"
						class="form-input h-full w-1/2 truncate border-0 border-b-[1px] border-[#ddd] bg-transparent py-2 pl-0 pr-4 text-sm focus-visible:ring-0"
						placeholder="Content"
						bind:value={newTagValue}
					/>
				</div>
				<button
					class="rounded-lg bg-[#c4c4c410] p-2 text-lg text-[#2D9CDB] hover:border active:bg-[#fff]"
					on:click={() => handleAddTag()}>Add</button
				>
			</div>

			<Scrollbar
				classLayout="flex flex-col items-center gap-2"
				className="max-h-36 overflow-auto"
			>
				{#each imgInfo.tag_list as [key, value], idx}
					<InfoTag
						on:delete={() => handleDeleteTag(idx)}
						on:edit={handleEditTag}
						tagKey={key}
						tagValue={value}
						{idx}
					/>
				{/each}
			</Scrollbar>
		</div>
	</Card>
</dialog>

<Modal bind:open={popupModal} size="xs" autoclose>
	<div class="text-center">
		<Icon
			name="exclamation-circle-outline"
			class="mx-auto mb-4 h-12 w-12 text-gray-400"
		/>
		<h3 class="mb-5 text-lg font-normal text-gray-500 dark:text-gray-400">
			Confirm delete this photo?
		</h3>
		<Button
			color="red"
			class="mr-2"
			on:click={() => {
				deleteImage(deleteImg);
			}}>Yes, I'm sure</Button
		>
		<Button color="alternative">No, cancel</Button>
	</div>
</Modal>

<style>
	.showModalAlbum {
		display: block;
	}
	.showClassifyAlbum {
		display: none;
	}

	@media only screen and (min-width: 1000px) {
		.showModalAlbum {
			display: none;
		}
		.showClassifyAlbum {
			display: block;
		}
	}
</style>
