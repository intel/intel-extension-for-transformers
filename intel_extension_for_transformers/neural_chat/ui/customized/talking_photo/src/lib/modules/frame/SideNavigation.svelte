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
	import { open } from "$lib/shared/stores/common/Store";
	import SidenavItems from "$lib/modules/frame/SidenavItems.svelte";
	import SidenavHeader from "$lib/modules/frame/SidenavHeader.svelte";
	import { clickOutside } from "$lib/network/sidebar/ClickOutside";

	const style = {
		mobilePosition: {
			left: "left-0 ",
			right: "hidden lg:left-0",
		},
		container: `pb-32 lg:pb-12`,
		close: `duration-700 ease-out hidden transition-all sm:w-24 shadow`,
		default: `bg-[#252e47] h-full overflow-y-auto top-0 sm:absolute sm:block sm:z-40 border-r border-[#252e47] hidden lg:left-0`,
		open: `absolute duration-500 ease-in transition-all  z-40 sm:w-64 shadow`,
	};

	
	export let mobilePosition = "right";
</script>

<aside
	use:clickOutside
	class={`${style.default} ${style.mobilePosition[mobilePosition]}
       ${$open = style.open} scrollbar`}
>
	<div class={style.container}>
		<SidenavHeader />
		<SidenavItems />
	</div>
</aside>

<style>
	.scrollbar::-webkit-scrollbar {
		width: 0;
		background: transparent; /* hide sidenav scrollbar on Chrome, Opera and other webkit Browsers*/
	}
	.scrollbar {
		-ms-overflow-style: none;
	}
</style>
