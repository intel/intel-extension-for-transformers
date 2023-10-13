<script>
	import "tailwindcss/tailwind.css";
	import "../app.postcss";
	import Notifications from "svelte-notifications";
	import Layout from "$lib/modules/frame/Layout.svelte";
	import { onMount } from "svelte";
	import Timer from "$lib/modules/settings/Timer.svelte";
	import Marquee from "$lib/modules/layout/Marquee.svelte";

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
			<Marquee />
			<Timer />
			<div class="h-0 grow">
				<slot />
			</div>
		</div>
	</Layout>
</Notifications>
