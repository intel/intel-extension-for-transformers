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
	import "tailwindcss/tailwind.css";
	import "../app.postcss";
	import Notifications from "svelte-notifications";
	import Layout from "$lib/modules/frame/Layout.svelte";
	import { onMount } from "svelte";

	onMount(() => {
		window.deviceType = window.innerWidth > 640 ? "pc" : "mobile";
		window.onresize = () => {
			window.deviceType = window.innerWidth > 640 ? "pc" : "mobile";
		};
		window.addEventListener("load", function () {
			setTimeout(function () {
				// This hides the address bar:
				window.scrollTo(0, 1);
			}, 0);
		});

	});
</script>

<Notifications>
	<Layout>
		<div class="flex h-full flex-col">
			<div class="h-0 grow bg-white  lg:rounded-tl-3xl">
				<slot />
			</div>
		</div>

	</Layout>
</Notifications>
