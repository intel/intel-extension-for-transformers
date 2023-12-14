<script>
	import { showSidePage } from "$lib/shared/stores/common/Store";
	import { clickOutside } from "$lib/network/sidebar/ClickOutside";
	import ChatBot from "$lib/modules/side-page/ChatBot.svelte";

	const style = {
		default: `fixed right-0 top-0 h-screen bg-[#f8f8f8] transition-all overflow-hidden`,
		close: `duration-200 ease-out w-0 shadow`,
		open: `max-sm:w-[90vw] duration-200 ease-in z-40 shadow`,
	};

</script>

<aside
	use:clickOutside
	on:click_outside={() => showSidePage.set(false)}
	class={`${style.default} ${$showSidePage ? style.open : style.close}`}
>
	<ChatBot />
</aside>
