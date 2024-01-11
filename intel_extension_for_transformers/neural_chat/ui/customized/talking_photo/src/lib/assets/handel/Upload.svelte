<script>
    import uploadFile from '$lib/assets/voice/svg/uploadFile.svg';
	export let type="";
</script>

<label for={type} class="flex flex-col items-center justify-center shadow-[0_2px_30px_0_rgba(0,0,0,0.1)] rounded-xl text-center  sm:w-[5rem] sm:h-[5rem] ">
	<img class="h-10 swap-on" src={uploadFile} alt=""/>
	<p class="text-xs text-gray-500">Upload</p>
</label>