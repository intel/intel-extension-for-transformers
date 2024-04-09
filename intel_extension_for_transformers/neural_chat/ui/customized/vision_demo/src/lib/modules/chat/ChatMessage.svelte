<script lang="ts">
	import MessageAvatar from "$lib/modules/chat/MessageAvatar.svelte";
	import type { Message } from "$lib/shared/constant/Interface";
	import { createEventDispatcher } from "svelte";

	let dispatch = createEventDispatcher();

	export let msg: Message;
	export let time: string = "";
</script>

<div
	class={msg.role === 0
		? "flex w-full gap-3"
		: "flex w-full items-center gap-3"}
>
	<div
		class={msg.role === 0
			? "flex aspect-square w-[0.16rem]   items-center justify-center rounded bg-[#10748f] max-sm:hidden"
			: "flex aspect-square h-10 w-[0.28rem]  items-center justify-center rounded bg-[#00c7fd] max-sm:hidden"}
	>
		<MessageAvatar role={msg.role} />
	</div>
	<div class="group relative items-center text-white">
		<div>
			<p
				class={msg.role === 0
					? "max-w-[60vw]  items-center  whitespace-pre-line break-keep text-[1rem] font-light leading-5 leading-relaxed sm:max-w-[50rem]"
					: "font-intel-medium max-w-[60vw]  items-center whitespace-pre-line break-keep text-[1rem] leading-5 leading-relaxed sm:max-w-[50rem]"}
			>
				{@html msg.content}
			</p>
		</div>
	</div>
</div>


<style>
	.wrap-style {
		word-wrap: break-word;
		word-break: break-all;
	}
</style>
