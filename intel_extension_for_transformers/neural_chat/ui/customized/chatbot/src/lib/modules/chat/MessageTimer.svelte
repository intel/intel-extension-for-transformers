<script lang="ts">
	export let time: string;
	import { createEventDispatcher } from "svelte";

	export let msecond_per_token = "0";
	export let first_token_latency = "0";
	let dispatch = createEventDispatcher();
</script>

<div class="flex justify-between">
	<div class="mb-4 flex items-center justify-start gap-1 space-x-2">
		<div class="items-center mt-[0.2rem]">
			<svg t="1712126546631" class="icon w-5 h-5" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="18158" width="64" height="64"><path d="M481.28 801.792h61.44v77.824h-61.44zM798.72 481.28h77.824v61.44H798.72zM136.192 481.28h77.824v61.44h-77.824z" fill="#63F7DE" p-id="18159"></path><path d="M512 955.392C267.264 955.392 68.608 756.736 68.608 512c0-125.952 54.272-246.784 147.456-330.752l54.272 61.44C193.536 311.296 150.528 409.6 150.528 512c0 199.68 162.816 361.472 361.472 361.472S873.472 711.68 873.472 512C873.472 274.432 675.84 150.528 481.28 150.528v-81.92c235.52 0 474.112 152.576 474.112 443.392 0 244.736-198.656 443.392-443.392 443.392z" fill="#437DFF" p-id="18160"></path><path d="M481.28 75.776h81.92v443.392h-81.92z" fill="#437DFF" p-id="18161"></path><path d="M499.67104 468.7872l63.6928 51.52768-108.83072 134.5536-63.6928-51.52768z" fill="#437DFF" p-id="18162"></path></svg>
			<!-- <svg t="1712126103018" class="icon w-5 h-5" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4272" width="16" height="16"><path d="M512 967.111111a455.111111 455.111111 0 1 1 455.111111-455.111111 455.623111 455.623111 0 0 1-455.111111 455.111111zM512 147.911111A364.088889 364.088889 0 1 0 876.088889 512 364.487111 364.487111 0 0 0 512 147.911111z m227.555556 455.111111H420.977778V284.444444H512v227.555556h227.555556v91.022222z" fill="#1B7CFF" p-id="4273"></path></svg> -->
		</div>
		{#if msecond_per_token !== "0" && first_token_latency !== "0"}
			<div class="flex items-center space-x-1 text-sm text-gray-800">
				<strong>First Token Latency: </strong>
				<p>{first_token_latency}</p>
			</div>
			<div
				class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
			/>
			<div class="flex items-center space-x-1 text-sm text-gray-800">
				<strong>Average Token Latency: </strong>
				<p>{msecond_per_token}</p>
			</div>
			<div
				class="inline-block w-0.5 self-stretch bg-gray-300 opacity-100 dark:opacity-50"
			/>
			<div class="flex items-center space-x-1 text-sm text-gray-800">
				<strong>End to End Time: </strong>
				<p>{time}s</p>
			</div>
		{/if}
	</div>

	<div class="fixed right-10 bottom-10 items-center justify-end gap-2 space-x-2">
		<svg 
		on:click={() => {
			dispatch("handleTop");
		}}
		t="1712126902850" class="icon w-10  h-10 hover:opacity-80" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="22708" width="64" height="64"><path d="M512 512m-448 0a448 448 0 1 0 896 0 448 448 0 1 0-896 0Z" fill="#92C8FF" p-id="22709"></path><path d="M749.344 493.347c-24.884 24.885-65.602 24.885-90.485 0l-82.855-82.859v356.516c0 35.2-28.8 64-64 64s-64-28.8-64-64V410.488l-82.855 82.859c-24.883 24.885-65.602 24.885-90.485 0-24.881-24.882-24.881-65.599 0-90.482l189.28-189.291a64.115 64.115 0 0 1 2.72-2.915c12.466-12.467 28.907-18.686 45.34-18.661 16.433-0.025 32.874 6.194 45.34 18.661a64.115 64.115 0 0 1 2.72 2.915l189.28 189.291c24.881 24.883 24.881 65.6 0 90.482z" fill="#409EFF" p-id="22710"></path></svg>	
		
		<!-- <svg
				xmlns="http://www.w3.org/2000/svg"
				xml:space="preserve"
				viewBox="0 0 21.6 21.6"
				width="24"
				height="24"
				class="w-5 fill-[#0597ff] hover:fill-[#0597ff]"
				on:click={() => {
					dispatch("handleTop");
				}}
				><path
					d="M2.2 3.6V.8h17.2v2.8zm7.2 17.2V10.4L5.8 14l-1.9-1.9 6.9-6.9 6.9 6.9-1.9 1.9-3.6-3.6v10.4z"
				/></svg
			> -->

	</div>
</div>
