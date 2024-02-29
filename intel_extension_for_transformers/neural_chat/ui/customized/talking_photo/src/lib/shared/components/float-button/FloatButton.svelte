<script lang="ts">
	import { goto } from "$app/navigation";

	export let destination: string;
	export let url: string;
	const onClick = () => {
		goto(url);
	};
</script>

<!-- class="mt-16 ml-4 h-10 w-24 rounded-md bg-gradient-to-r from-cyan-500 to-blue-500 text-sm text-[#eee] shadow-inner max-sm:fixed max-sm:bottom-10 max-sm:right-10 max-sm:h-10 max-sm:w-10 max-sm:rounded-full max-sm:text-xs" -->

<button
	on:click={onClick}
	class="rounded-lg bg-indigo-600 px-4 py-3 text-center text-[0.8rem] font-semibold text-white shadow-md transition duration-200 ease-in hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-indigo-200 max-sm:p-2"
>
	{destination}
</button>
