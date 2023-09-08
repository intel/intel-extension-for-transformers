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
			<!-- <div class="flex items-center justify-end ml-5 p-1 relative w-full sm:mr-0 sm:right-auto">
				<span class="block pr-5">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-5 text-gray-600 w-5"
						viewBox="0 0 20 20"
						fill="currentColor"
					>
						<path
							fill-rule="evenodd"
							d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z"
							clip-rule="evenodd"
						/>
					</svg>
				</span>
				<span class="block pr-5">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-6 w-6 text-gray-600"
						viewBox="0 0 20 20"
						fill="currentColor"
					>
						<path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
						<path
							d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z"
						/>
					</svg>
				</span>
				<span class="block pr-5 relative">
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="h-6 text-gray-600 w-6"
						viewBox="0 0 20 20"
						fill="currentColor"
					>
						<path
							fill-rule="evenodd"
							d="M4.083 9h1.946c.089-1.546.383-2.97.837-4.118A6.004 6.004 0 004.083 9zM10 2a8 8 0 100 16 8 8 0 000-16zm0 2c-.076 0-.232.032-.465.262-.238.234-.497.623-.737 1.182-.389.907-.673 2.142-.766 3.556h3.936c-.093-1.414-.377-2.649-.766-3.556-.24-.56-.5-.948-.737-1.182C10.232 4.032 10.076 4 10 4zm3.971 5c-.089-1.546-.383-2.97-.837-4.118A6.004 6.004 0 0115.917 9h-1.946zm-2.003 2H8.032c.093 1.414.377 2.649.766 3.556.24.56.5.948.737 1.182.233.23.389.262.465.262.076 0 .232-.032.465-.262.238-.234.498-.623.737-1.182.389-.907.673-2.142.766-3.556zm1.166 4.118c.454-1.147.748-2.572.837-4.118h1.946a6.004 6.004 0 01-2.783 4.118zm-6.268 0C6.412 13.97 6.118 12.546 6.03 11H4.083a6.004 6.004 0 002.783 4.118z"
							clip-rule="evenodd"
						/>
					</svg>
				</span>
				
			</div> -->
		</div>
	</div>
</header>