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
	import IntelLogo from "$lib/assets/header/intelLogo.svelte";
</script>

<header class="sticky top-0 z-40 mx-auto w-full flex-none bg-[#1f2029] pl-[2.2rem] pr-[3.5rem]">
	<nav
		class="w-full divide-gray-200 border-gray-200 bg-[#1f2029] p-0 m-0 items-center text-gray-500 dark:divide-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400"
	>
		<div class="mx-auto flex w-full flex-wrap items-center justify-between p-0 m-0">
			<IntelLogo />
			<span
				class="self-center whitespace-nowrap text-[1.45rem] font-semibold text-white p-0 m-0"
				data-svelte-h="svelte-1hbktnk">ChatBot</span
			>
		</div>
	</nav>
</header>
