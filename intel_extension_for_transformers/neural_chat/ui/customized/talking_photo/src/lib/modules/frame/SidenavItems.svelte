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
	import { data } from '$lib/shared/constant/Data';
	import { page } from '$app/stores';
	import { open } from '$lib/shared/stores/common/Store';

	const style = {
		title: `mx-4 text-sm`,
		inactive: `hover:bg-[#252e47]`,
		active: `font-medium bg-[#ffffff1a] hover:bg-[#ffffff1a] text-[#ffffffe6] hover:text-[#ffffffe6] rounded-lg`,
		link: `flex items-center justify-start  p-6 w-full text-white  my-3`,
		close: `sm:duration-700 sm:ease-out sm:invisible sm:opacity-0 sm:transition-all`,
		open: `sm:duration-500 sm:ease-in sm:h-auto sm:opacity-100 sm:transition-all sm:w-auto`
	};
</script>

<ul class="sm:pl-3 pr-3">
	<li class="">
		{#each data as item, i (item.title)}
			<a
				href={item.link}
				class={`${style.link} 
            	${item.link === $page.url.pathname ? style.active : style.inactive}`}
			>
				<span><svelte:component this={item.icon} /></span>
				<span class={`${style.title} ${$open ? style.open : style.close}`}>
					{item.title}
				</span>
			</a>

		{/each}
	</li>
</ul>