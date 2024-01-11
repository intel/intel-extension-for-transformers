<script>
	import HeaderIcon from "$lib/assets/frame/svelte/headerTitle.svelte";
	import { data } from "$lib/shared/constant/Data";

	import {
		Button,
		Dropdown,
		DropdownDivider,
		DropdownItem,
	} from "flowbite-svelte";
	import { page } from "$app/stores";
	import ChatIcon from "$lib/assets/layout/svelte/ChatIcon.svelte";
	import BotIcon from "$lib/assets/layout/svelte/BotIcon.svelte";
	import { showSidePage } from "$lib/shared/stores/common/Store";
	import SettingIcon from "$lib/assets/chat/svelte/Setting.svelte";
	import { goto } from "$app/navigation";
	import AlbumIcon from "$lib/assets/layout/svelte/AlbumIcon.svelte";
	import HeaderSetting from "$lib/assets/chat/svelte/HeaderSetting.svelte";

	$: activeUrl = $page.url.pathname;
	let activeClass =
		"text-green-700 dark:text-green-300 hover:text-green-900 dark:hover:text-green-500";
</script>

<!-- <nav
	class="minimum-width-rem shadow-xs svelte-1482j9b sticky inset-x-0 top-0 w-full border-b bg-white delay-500 sm:hidden sm:h-[112px]"
>
	<div
		class="flex h-14 w-full items-center justify-between gap-4 px-3 sm:h-20 sm:px-10"
	>
		<div class="flex items-center gap-4">
			<HeaderIcon />
			<p class="font-bold">Talking with your photos</p>
		</div>
	</div>
</nav> -->

<header
	class="fixed left-0 top-0 z-30 flex w-full flex-row items-center justify-between bg-white px-4 shadow-md sm:hidden"
>
	<a class="flex flex-row items-center justify-center" href="/"
		><div
			class="logo svelte-1a6f506 mr-3 flex h-12 w-12 items-center justify-center"
		>
			<HeaderIcon />
		</div>
		<!-- <h1 class="stroke-orange-50 text-xl font-semibold leading-none">
			Neural Chat
		</h1> -->
		<div class="flex flex-col">
			<p class="font-bold">NeuralChat</p>
			<span class="text-[0.75rem] text-gray-600">Create Your Own Chatbot</span>
		</div>
	</a>

	<div class="right-0 flex flex-row gap-5">
		<button
			on:click={() => {
				showSidePage.set(true);
				goto("/");
			}}
		>
			<BotIcon />
		</button>
		<button
			on:click={() => {
				goto("/classify");
			}}
		>
			<AlbumIcon />
		</button>
		<button>
			<a href="/settings">
				<HeaderSetting />
			</a>
		</button>
	</div>

	<!-- <Button
		color="light"
		outline={true}
		class="border-none m-2 text-3xl active:border-none nav-btn">≡</Button
	>
	<Dropdown {activeUrl}>
		{#each data as item, i (item.title)}
			<DropdownItem href={item.link}>{item.title}</DropdownItem>
			<DropdownDivider />
		{/each}
	</Dropdown> -->
</header>
