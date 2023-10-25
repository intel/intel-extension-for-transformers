<script lang="ts">
	import { handleImageUpload } from "$lib/network/image/UploadImage"
	import UploadImageBlobs from "$lib/shared/components/upload/UploadImageBlobs.svelte";
	import { Progressbar } from "flowbite-svelte";

	let uploadProgress = 0
	let uploadHandle: number

	function handleUploadBegin() {
		uploadHandle = setInterval(() => {
			if (uploadProgress < 70) uploadProgress += 5
			else if (uploadProgress < 90) uploadProgress += 2
			else if (uploadProgress < 99) uploadProgress += 1
		}, 500);
	}

	function handleUploadEnd() {
		uploadProgress = 0
		clearInterval(uploadHandle)
	}

	function handleUploadClick(e) {
        new Promise(resolve => {
            handleImageUpload(e, resolve);
        }).then(() => {
            handleUploadEnd()
        })
        handleUploadBegin()
    }
</script>

<div class="relative grid grid-cols-1 mb-10 md:grid-cols-2 lg:my-14 lg:grid-cols-5">
	{#if uploadProgress}
		<Progressbar progress={uploadProgress.toString()} size='h-1' color='blue' divClass='absolute top-0 w-full lg:-top-14' />		
	{/if}
	<div class="col-span-3 rounded-l-lg bg-white p-4 lg:mx-14 sm:p-24 md:rounded-l-2xl">
		<div class="lg:flex-shrink-1 bg-white px-6 py-8 dark:bg-gray-800 lg:p-12">
			<h4
				class="flex-shrink-0 bg-white pr-4 text-sm font-semibold uppercase leading-5 tracking-wider text-indigo-600 dark:bg-gray-800"
			>
				Your own Chat
			</h4>
			<h3
				class="mt-5 text-2xl font-extrabold leading-8 text-gray-900 dark:text-white sm:text-3xl sm:leading-9"
			>
				Create Your Own Chat
			</h3>
			<p class="mt-6 text-base leading-6 text-gray-500 dark:text-gray-200">
				Based on the image file you uploaded, generate the corresponding
				conversation scene and create your own bot.
			</p>
			<div class="text-left pt-10 sm:hidden">
				<UploadImageBlobs on:upload={handleUploadClick}>
					<span class="bg-[#254ACE] text-white rounded py-2 px-4">Upload Images ↑</span>
				</UploadImageBlobs>
			</div>
		</div>
	</div>
	<div class="col-span-2 rounded-r-lg bg-white flex flex-col justify-center p-4 sm:p-6 sm:pr-20 md:rounded-r-2xl">
		<div class="grid grid-cols-3 gap-4">
			<UploadImageBlobs on:upload={handleUploadClick} />
		</div>
	</div>
</div>
