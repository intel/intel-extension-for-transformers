<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

<script lang="ts">
	import MessageAvatar from "$lib/assets/icons/message-avatar.svelte";
	import MessageTimer from "./message-timer.svelte";
	import ChatAudio from "./chat-audio.svelte";
	import { MESSAGE_ROLE } from "$lib/components/shared/shared.type";
		
	export let type: string;
	export let message: string | string[];
	export let text: string = ''
	export let displayTimer: Boolean = true;

	let isbot = (type === MESSAGE_ROLE.ASSISTANT || type === MESSAGE_ROLE.SYSTEM)
	let isuser = (type === MESSAGE_ROLE.HUMAN || type === MESSAGE_ROLE.USER)

	let playIdx = 0;
	let autoPlay = message.length > 0 && message[message.length - 1] !== "done";
	function handlePlayEnded() {
		playIdx++;
		autoPlay = true;
		if (playIdx < message.length && message[playIdx] === 'done') {
			playIdx = 0;
			autoPlay = false;
		}
	}
</script>

<div
	class="flex w-full mt-4 space-x-3 {type === 'Human' || type === 'user'
		? 'ml-auto justify-end'
		: ''}"
>
	{#if isbot}
		<div class="flex justify-center items-center h-[47px] w-[47px] rounded bg-[#0068B5]">
			<MessageAvatar role={MESSAGE_ROLE.ASSISTANT } />
		</div>
	{/if}
	<div class="relative group">
		<div
			class={isuser ? "bg-blue-600 text-white p-3 rounded-l-lg rounded-br-lg wrap-style"
							: "border-2 p-3 rounded-r-lg rounded-bl-lg wrap-style"}
		>
			{#if Array.isArray(message)}
				{#key playIdx}
					<ChatAudio src={message[playIdx]} {autoPlay} {text} on:ended={handlePlayEnded} right/>
				{/key}
			{:else if message.includes("blob:")}
				<ChatAudio src={message} {text}/>
			{:else}
				<p class="text-sm message max-w-md line">{message}</p>
			{/if}
		</div>
		{#if displayTimer}
			<MessageTimer />
		{/if}
	</div>
	{#if isuser}
		<div class="flex justify-center items-center h-[47px] w-[47px] rounded bg-[#45b7f3]">
			<MessageAvatar role={MESSAGE_ROLE.USER} />
		</div>
	{/if}
</div>

<style>
	.wrap-style {
		width: 100%;
		height: auto;
		word-wrap: break-word;
		word-break: break-all;
	}

	audio::-webkit-media-controls-panel {
		background-color: rgb(37 99 235);
	}

</style>
