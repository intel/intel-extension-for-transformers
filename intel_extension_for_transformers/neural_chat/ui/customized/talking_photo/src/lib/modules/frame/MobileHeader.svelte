<script>
	import HeaderIcon from "$lib/assets/frame/svelte/headerTitle.svelte";
	import { data } from "$lib/shared/constant/Data";

	import { Button, Dropdown, DropdownDivider, DropdownItem } from "flowbite-svelte";
	import { page } from "$app/stores";
	$: activeUrl = $page.url.pathname;
	let activeClass =
		"text-green-700 dark:text-green-300 hover:text-green-900 dark:hover:text-green-500";
</script>

<header
	class="fixed left-0 top-0 z-30 flex w-full flex-row items-center justify-between bg-white px-4 shadow-md sm:hidden"
>
	
	<a class="flex flex-row items-center justify-center" href="/"
		><div
			class="logo svelte-1a6f506 mr-3 flex h-12 w-12 items-center justify-center"
		>
			<HeaderIcon />
		</div>
		<p class="font-bold">TalkingPhotos</p>

		</a
	>
	<Button
		color="light"
		outline={true}
		class="border-none m-2 text-3xl active:border-none nav-btn">≡</Button
	>
	<Dropdown {activeUrl}>
		{#each data as item, i (item.title)}
			<DropdownItem href={item.link}>{item.title}</DropdownItem>
			<DropdownDivider />

		{/each}
	</Dropdown>

</header>
