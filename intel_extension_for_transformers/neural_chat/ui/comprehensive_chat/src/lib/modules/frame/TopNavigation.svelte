<script>
	import { newUploadNum, open } from "$lib/shared/stores/common/Store";
	import { data } from "$lib/shared/constant/Data";
	import { page } from "$app/stores";
	import MobileHeader from "$lib/modules/frame/MobileHeader.svelte";
	import { Indicator } from "flowbite-svelte";

	const style = {
		title: `mx-4 text-sm`,
		inactive: `text-[#27272a]`,
		active: `font-medium text-blue-400`,
		link: `flex flex-col items-center justify-start p-4 w-full`,
		close: `lg:duration-700 lg:ease-out lg:invisible lg:opacity-0 lg:transition-all`,
		open: `lg:duration-500 lg:ease-in lg:h-auto lg:opacity-100 lg:transition-all lg:w-auto`,
	};

</script>

<header class="frame-header z-10 w-full items-center bg-white shadow sm:relative md:h-20">
	<div
		class="group relative hidden h-full w-full items-center gap-5 px-3 pr-5 justify-end sm:flex "
	>
		<span class="font-bold bg-gradient-to-r from-sky-500 to-indigo-500 bg-clip-text text-transparent">Upload Pictures, Start an Amazing Journey!</span>
	</div>
	<MobileHeader />
</header>
