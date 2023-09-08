<script>
	import data from './data';
	import { page } from '$app/stores';
	import { open } from '$lib/components/shared/store';

	const style = {
		title: `mx-4 text-sm`,
		inactive: `text-white hover:text-blue-200`,
		active: `font-medium text-blue-300 hover:text-blue-300`,
		link: `flex items-center justify-start my-2 p-4 w-full`,
		close: `lg:duration-700 lg:ease-out lg:invisible lg:opacity-0 lg:transition-all`,
		open: `lg:duration-500 lg:ease-in lg:h-auto lg:opacity-100 lg:transition-all lg:w-auto`
	};
</script>

<ul class="md:pl-5">
	<li>
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