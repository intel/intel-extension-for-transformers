<script lang="ts">
	import MessageAvatar from "./MessageAvatar.svelte";
	import MessageTimer from "./MessageTimer.svelte";
	import LikeButtonIcon from "$lib/assets/likeButtonIcon.svelte";
	import DislikeButtonIcon from "$lib/assets/dislikeButtonIcon.svelte";
	import TextToImgButtonIcon from "$lib/assets/text2imgButtonIcon.svelte";
	import { env } from '$env/dynamic/public';

	let clickLike = false;
	let clickDislike = false;
	let showImg = false;
	let imgPromise: Promise<any>;

	export let type: string;
	export let message: string;
	export let displayTimer: Boolean = true;

	const msg = "Welcome to Neural Chat! 😊";

	function typewriter(node: HTMLElement, { speed = 1 }) {
		const valid = (
			node.childNodes.length === 1 &&
			node.childNodes[0].nodeType === Node.TEXT_NODE
		);

		if (!valid) {
			throw new Error(`This transition only works on elements with a single text node child`);
		}

		const text = node.textContent;
		const duration = text.length / (speed * 0.01);

		return {
			duration,
			tick: t => {
				const i = Math.trunc(text.length * t);
				node.textContent = text.slice(0, i);
			}
		}
	};

	async function handleImgClick() {
		if (showImg == true) {
			showImg = false;
			return;
		}
		const url = env.TXT2IMG;

		imgPromise = (async () => {
			const init: RequestInit = {
				method: "POST",
				// mode: 'no-cors',
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					prompt: message,
					steps: 25,
					guidance_scale: 7.5,
					seed: 42,
					token: "intel_sd_bf16_112233",
				}),
			};
			let response = await fetch(url, init);
			return (await response.json()).img_str;
		})();

		showImg = true;
	}
</script>

<div
	class="flex w-full mt-2 space-x-3 {type === 'Human' || type === 'user'
		? 'ml-auto justify-end'
		: ''}"
>
	{#if type === "Assistant" || type === "assistant" || type === "system"}
		<MessageAvatar role="Assistant" />
	{/if}
	<div class="relative group">
		<div
			class={type === "Human" || type === "user"
				? "bg-blue-600 text-white p-3 rounded-l-lg rounded-br-lg wrap-style"
				: "bg-gray-300 p-3 rounded-r-lg rounded-bl-lg wrap-style"}
		>
			{#if (type === "Assistant" || type === "assistant") && message == "Loading.."}
				<button class="btn loading btn-ghost text-blue-500 " />
			{:else if type === "assistant" || type === "Assistant"}
				<p
					in:typewriter
					class="text-sm message max-w-md line"
				>{message}</p>
			{:else}
			<p
				class="text-sm message max-w-md line"
			>{message}</p>
			{/if}
		</div>
		<!-- && message !== msg  -->
		{#if (type === "Assistant" || type === "assistant") && message !== "Loading.."}
			<div class="absolute h-5 -top-5 right-0 hidden group-hover:flex ">
				<!-- svelte-ignore a11y-click-events-have-key-events -->
				<figure
					class="w-5 h-5 opacity-40 text-black cursor-pointer hover:opacity-100 hover:text-yellow-600"
					class:opacity-100={clickLike}
					class:text-yellow-600={clickLike}
					on:click={() => {
						[clickLike, clickDislike] = [true, false];
					}}
				>
					<LikeButtonIcon />
				</figure>
				<!-- svelte-ignore a11y-click-events-have-key-events -->
				<figure
					class="w-5 h-5 opacity-40 text-black cursor-pointer hover:opacity-100 hover:text-yellow-600 rotate-180"
					class:opacity-100={clickDislike}
					class:text-yellow-600={clickDislike}
					on:click={() => {
						[clickDislike, clickLike] = [true, false];
					}}
				>
					<DislikeButtonIcon />
				</figure>
				<!-- svelte-ignore a11y-click-events-have-key-events -->
				<figure
					class="w-5 h-5 opacity-40 text-black cursor-pointer hover:opacity-100 hover:text-yellow-600"
					on:click={handleImgClick}
					class:opacity-100={showImg}
					class:text-yellow-600={showImg}
				>
					<TextToImgButtonIcon />
				</figure>
			</div>
		{/if}
		{#if showImg}
			{#await imgPromise}
				<progress class="progress" />
			{:then src}
				<img
					class="w-26 h-26 mt-2 hover:scale-150"
					src="data:image/png;base64,{src}"
					alt=""
				/>
			{/await}
		{/if}
		{#if displayTimer}
			<MessageTimer />
		{/if}
	</div>
	{#if type === "Human" || type === "user"}
		<MessageAvatar role="Human" />
	{/if}
</div>

<style>
	.wrap-style {
		width: 100%;
		height: auto;
		word-wrap: break-word;
		word-break: break-all;
	}
</style>
