<script>
	import { open } from '$lib/components/shared/store';

	const toggle = () => {
		$open = !$open;
	};
</script>


<header class="bg-white h-16 items-center relative shadow w-full z-10 md:h-20">
	<div class="flex flex-center flex-col h-full justify-center mx-auto px-3 relative">
		<div class="flex items-center pl-1 relative w-full sm:ml-0 sm:pr-2 lg:max-w-68">
			<div class="flex left-0 relative w-3/4 items-center">
				<div class="flex group h-full items-center relative w-12">
					<button
						type="button"
						aria-expanded="false"
						aria-label="Toggle sidenav"
						on:click={toggle}
						class="text-4xl text-black focus:outline-none"
					>
						&#8801;
					</button> 
				</div>
				<p class="mt-2 font-bold text-xl">Neural Chat</p>
			</div>
		</div>
	</div>
</header>