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
	import { open } from '$lib/components/shared/store';
	import SidenavItems from './Items.svelte';
	import SidenavHeader from './Header.svelte';
	import { clickOutside } from '$lib/components/shared/click-outside';

	const style = {
		mobilePosition: {
			left: 'left-0 ',
			right: 'right-0 lg:left-0'
		},
		container: `pb-32 lg:pb-12`,
		close: `duration-700 ease-out hidden transition-all lg:w-24`,
		default: `bg-[#161730] h-screen overflow-y-auto top-0 lg:absolute lg:block lg:z-40`,
		open: `absolute duration-500 ease-in transition-all w-8/12 z-40 sm:w-5/12 md:w-64`
	};

	const closeSidenav = () => {
		//close sidenav on click outside when viewport is less than 1024px
		$open = false;
	};
	export let mobilePosition = 'right';
</script>

<aside
	use:clickOutside
	on:click_outside={closeSidenav}
	class={`${style.default} ${style.mobilePosition[mobilePosition]}
       ${$open ? style.open : style.close} scrollbar`}
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