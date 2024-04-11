<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

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

<header class="frame-header z-10 w-full items-center bg-[#252e47] shadow sm:relative md:h-20">
	<div
		class="group relative hidden h-full w-full items-center gap-5 px-3 pr-5 justify-end sm:flex "
	>
		<span class="font-bold bg-gradient-to-r from-sky-100 to-indigo-500 bg-clip-text text-transparent">Upload Your Photos, Start an Amazing Journey!</span>
	</div>
	<MobileHeader />
</header>
