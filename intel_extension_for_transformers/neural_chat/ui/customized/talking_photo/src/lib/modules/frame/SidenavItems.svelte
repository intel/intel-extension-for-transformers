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