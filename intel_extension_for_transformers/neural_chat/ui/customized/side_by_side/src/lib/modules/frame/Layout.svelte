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

<script lang="ts">
	import { onMount } from "svelte";
	import { page } from "$app/stores";
	import { browser } from "$app/environment";
	import { open } from "$lib/shared/stores/common/Store";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";

	let root: HTMLElement
	onMount(() => {
		document.getElementsByTagName("body").item(0)!.removeAttribute("tabindex");
		// root.style.height = document.documentElement.clientHeight + 'px'
	});

	if (browser) {
		page.subscribe(() => {
			// close side navigation when route changes
			if (window.innerWidth > 768) {
				$open = true;
			}
		});
	}
</script>

<div bind:this={root} class='h-full overflow-hidden relative'>
	<div class="h-full flex items-start">
		<div class='relative flex flex-col h-full pl-0 w-full  bg-white'>
			<Scrollbar className="h-0 grow " classLayout="h-full" alwaysVisible={false}>
				<slot />
			</Scrollbar>
		</div>
	</div>
</div>
