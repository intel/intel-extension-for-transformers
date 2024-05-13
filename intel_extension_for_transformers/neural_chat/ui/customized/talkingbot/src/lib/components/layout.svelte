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
	import { onMount } from 'svelte';
	import { page } from '$app/stores';
	import { browser } from '$app/environment';
	import { open } from '$lib/components/shared/store';
	import Overlay from '$lib/components/shared/overlay.svelte';
	import TopNavigation from '$lib/components/topnavigation/Index.svelte';
	import SideNavigation from '$lib/components/sidenavigation/Index.svelte';

	const style = {
		container: `bg-gray-100 h-screen overflow-hidden relative`,
		main: `h-screen overflow-auto pb-36 pt-4 px-2 md:pb-8 lg:px-4`,
		mainContainer: `flex flex-col h-screen pl-0 w-full lg:pl-24 lg:space-y-4`
	};

	onMount(() => {
		document.getElementsByTagName('body').item(0).removeAttribute('tabindex');
	});

	if (browser) {
		page.subscribe(() => {
			// close side navigation when route changes
			$open = false;
		});
	}
</script>

<div class={style.container}>
	<div class="flex items-start">
		<Overlay />
		<SideNavigation mobilePosition="right" />
		<div class={style.mainContainer}>
			<TopNavigation />
			<main class={style.main}>
				<slot />
			</main>
		</div>
	</div>
</div>
