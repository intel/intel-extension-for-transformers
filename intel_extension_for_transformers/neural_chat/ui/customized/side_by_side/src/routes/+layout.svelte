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
