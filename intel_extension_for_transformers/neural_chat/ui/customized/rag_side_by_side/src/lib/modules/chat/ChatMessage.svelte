<script lang="ts">
	import MessageAvatar from "$lib/modules/chat/MessageAvatar.svelte";
	import { type Message } from "$lib/shared/constant/Interface";

	export let msg: Message;
</script>

<div
class={msg.role === 0
	? "flex w-full gap-3"
	: "flex w-full gap-3 items-center"}
	
>
	<div
		class={msg.role === 0
			? "flex aspect-square w-[1px]  items-center justify-center rounded bg-[#0068B5] max-sm:hidden"
			: "flex aspect-square h-10 w-[1px] items-center justify-center rounded bg-[#000] max-sm:hidden"}
	>
		<MessageAvatar role={msg.role} />
	</div>
	<div class="group relative items-center">
		<div>
			<p
				class="max-w-[60vw] items-center whitespace-pre-line break-keep text-[0.8rem] leading-5 sm:max-w-[32rem]"
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
