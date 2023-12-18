<script lang="ts">
	import MessageAvatar from "$lib/modules/chat/MessageAvatar.svelte";
	import MessageTimer from "$lib/modules/chat/MessageTimer.svelte";
	import { MessageRole, type Message, MessageType } from "$lib/shared/constant/Interface";
	import ChatAudio from "$lib/modules/chat/ChatAudio.svelte";
	import { fromTimeStampToTime } from "$lib/shared/Utils";
	import ImageListMessage from "./ImageListMessage.svelte";
	import SingleImageMessage from "./SingleImageMessage.svelte";
	import AudioListMessage from "./AudioListMessage.svelte";
	import VideoMessage from './VideoMessage.svelte'

	export let msg: Message;
	export let time: string = ''

	const convertTypeImageList = (content: typeof msg.content) => content as {imgSrc: string;imgId: string;}[]
	const convertTypeAudioList = (content: typeof msg.content) => content as string[]
	const convertTypeSingleImage = (content: typeof msg.content) => content as {imgSrc: string;imgId: string;}
	const convertTypeString = (content: typeof msg.content) => content as string
</script>

{#if time}
	<MessageTimer {time} />
{/if}

<div
	class="flex w-full gap-3"
	class:flex-row-reverse={msg.role === MessageRole.User}
>
	<div
		class="flex aspect-square h-10 items-center justify-center rounded bg-[#0068B5] max-sm:hidden"
	>
		<MessageAvatar role={msg.role} />
	</div>
	<div class="group relative">
		<div
			class={msg.role === MessageRole.User
				? "wrap-style relative ml-4 rounded-l-lg rounded-br-lg border-2 border-[#3369FF] bg-[#3369FF] p-2 text-white"
				: "wrap-style relative mr-4 rounded-r-lg rounded-bl-lg  bg-blue-50 p-2 text-blue-800"}
		>
			{#if msg.type === MessageType.ImageList}
				<ImageListMessage content={convertTypeImageList(msg.content)} />
			{:else if msg.type === MessageType.SingleImage}
				<SingleImageMessage content={convertTypeSingleImage(msg.content)}/>
			{:else if msg.type === MessageType.AudioList}
				<AudioListMessage content={convertTypeAudioList(msg.content)} />
			{:else if msg.type === MessageType.SingleAudio}
				<ChatAudio src={convertTypeString(msg.content)} />
			{:else if msg.type === MessageType.singleVideo}
				<VideoMessage src={convertTypeString(msg.content)}/>
			{:else}
				<p class="sm:max-w-[32rem] max-w-[60vw] whitespace-pre-line text-[0.8rem] break-keep leading-5">{@html msg.content}</p>
			{/if}
		</div>
	</div>
</div>

<style>
	.wrap-style {
		word-wrap: break-word;
		word-break: break-all;
	}
</style>
