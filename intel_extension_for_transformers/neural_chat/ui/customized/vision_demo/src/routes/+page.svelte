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
	export let data;
	import {
		knowledge1,
		latencyWritable,
	} from "$lib/shared/stores/common/Store";
	import { onMount } from "svelte";
	import {
		LOCAL_STORAGE_KEY,
		MessageRole,
		MessageType,
		type Message,
	} from "$lib/shared/constant/Interface";
	import {
		getCurrentTimeStamp,
		scrollToBottom,
		scrollToTop,
	} from "$lib/shared/Utils";
	import { fetchTextStream } from "$lib/network/chat/Network";
	import LoadingAnimation from "$lib/shared/components/loading/Loading.svelte";
	import "driver.js/dist/driver.css";
	import "$lib/assets/layout/css/driver.css";
	import PaperAirplane from "$lib/assets/chat/svelte/PaperAirplane.svelte";
	import Clear from "$lib/assets/chat/svelte/ChatBot_Latency_Clear_button.svg";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import ChatMessage from "$lib/modules/chat/ChatMessage.svelte";
	import LantencyHint from "$lib/modules/lantencyHint/LantencyHint.svelte";

	let query: string = "";
	let loading: boolean = false;
	let scrollToDiv: HTMLDivElement;
	// ·········
	let chatMessages: Message[] = data.chatMsg ? data.chatMsg : [];
	console.log("chatMessages", chatMessages);

	// ··············

	$: knowledge_1 = $knowledge1?.id ? $knowledge1.id : "default";

	onMount(async () => {
		scrollToDiv = document
			.querySelector(".chat-scrollbar")
			?.querySelector(".svlr-viewport")!;
	});

	function handleTop() {
		console.log("top");

		scrollToTop(scrollToDiv);
	}

	function storeMessages() {
		console.log("localStorage", chatMessages);

		localStorage.setItem(
			LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY,
			JSON.stringify(chatMessages)
		);
	}

	function extractContent(data) {
		if (data.startsWith("b'") || data.startsWith('b"')) {
			data = data.slice(2);
		}
		if (data.endsWith("'") || data.endsWith('"')) {
			data = data.slice(0, -1);
		}

		if (data.includes("\\n")) {
			if (data === "\\n") {
				data = "";
			} else {
				data = data.replace(/\\n/g, "<br />");
			}
		}
		return data;
	}

	const callTextStream = async (query: string) => {
		const eventSource = await fetchTextStream(query, knowledge_1);
		console.log("eventSource", eventSource);

		eventSource.addEventListener("message", (e: any) => {
			let currentMsg = extractContent(e.data);
			console.log("currentMsg", currentMsg);

			currentMsg = currentMsg.replace("@#$", " ");
			if (currentMsg == "[DONE]") {
				console.log("done getCurrentTimeStamp", getCurrentTimeStamp);
				let startTime = chatMessages[chatMessages.length - 1].time;

				loading = false;
				let totalTime = parseFloat(
					((getCurrentTimeStamp() - startTime) / 1000).toFixed(2)
				);
				console.log("done totalTime", totalTime);
				console.log(
					"chatMessages[chatMessages.length - 1]",
					chatMessages[chatMessages.length - 1]
				);

				if (chatMessages.length - 1 !== -1) {
					chatMessages[chatMessages.length - 1].time = totalTime;
				}
				console.log("done chatMessages", chatMessages);

				storeMessages();
			} else {
				if (chatMessages[chatMessages.length - 1].role == MessageRole.User) {
					chatMessages = [
						...chatMessages,
						{
							role: MessageRole.Assistant,
							type: MessageType.Text,
							content: currentMsg,
							time: getCurrentTimeStamp(),
							first_token_latency: "0",
							msecond_per_token: "0",
						},
					];
					console.log("? chatMessages", chatMessages);
				} else {
					if (currentMsg.includes("first_token_latency")) {
						const matchResult = currentMsg.match(/(\d+(\.\d{1,2})?)/);
						const extractedNumber = parseFloat(matchResult[0]).toFixed(1);
						chatMessages[chatMessages.length - 1].first_token_latency =
							extractedNumber + " ms";
					} else if (currentMsg.includes("msecond_per_token")) {
						const matchResult = currentMsg.match(/(\d+(\.\d{1,2})?)/);
						const extractedNumber = parseFloat(matchResult[0]).toFixed(1);
						chatMessages[chatMessages.length - 1].msecond_per_token =
							extractedNumber + " ms";
						latencyWritable.set(extractedNumber);
					} else {
						let content = chatMessages[chatMessages.length - 1]
							.content as string;
						chatMessages[chatMessages.length - 1].content =
							content + currentMsg;
					}
				}
				scrollToBottom(scrollToDiv);
			}
		});
		eventSource.stream();
	};

	const handleTextSubmit = async () => {
		console.log("handleTextSubmit");

		loading = true;
		const newMessage = {
			role: MessageRole.User,
			type: MessageType.Text,
			content: query,
			time: 0,
		};
		chatMessages = [...chatMessages, newMessage];
		scrollToBottom(scrollToDiv);
		storeMessages();
		query = "";

		await callTextStream(newMessage.content);

		scrollToBottom(scrollToDiv);
		storeMessages();
	};

	function handelClearHistory() {
		localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY);
		chatMessages = [];
	}

</script>

<div
	class="flex flex-col grow"
>
	<div class="relative h-full items-center gap-5 bg-fixed sm:flex">
		<div
			class="relative h-full ml-[12%] flex  w-full flex-col sm:mt-0 sm:w-[66%]"
		>
			<div
				class="fixed relative flex w-full flex-col items-center justify-between px-2"
			>
				<div class="relative my-2 flex w-full flex-row justify-center">
					<div class="focus:border-none relative w-full">
						<input
							class="text-md block w-full border-0 border-b-2 border-gray-300 bg-[#183544] px-1
						py-4 text-white focus:border-gray-300 focus:ring-0"
							type="text"
							placeholder="Enter prompt here"
							disabled={loading}
							maxlength="1200"
							bind:value={query}
							on:keydown={(event) => {
								if (event.key === "Enter" && !event.shiftKey && query) {
									event.preventDefault();
									handleTextSubmit();
								}
							}}
						/>
						<button
							on:click={() => {
								if (query) {
									handleTextSubmit();
								}
							}}
							type="submit"
							class="absolute bottom-2.5 end-2.5 px-4 py-2 text-sm font-medium text-white dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
							><PaperAirplane /></button
						>
					</div>
				</div>
			</div>

			<!-- clear -->
			{#if Array.isArray(chatMessages) && chatMessages.length > 0 && !loading}
				<div class="flex w-full justify-between pr-5">
					<div class="flex items-center">
						<button
							class="bg-primary text-primary-foreground hover:bg-primary/90 group flex items-center justify-center space-x-2 p-2"
							type="button"
							on:click={() => handelClearHistory()}
						>
							<img class="ml-[0.3rem] w-[6rem]" src={Clear} alt="Your SVG" />
						</button>
					</div>
				</div>
			{/if}
			<!-- clear -->

			<div class="mx-auto flex h-full w-full flex-col">
				<Scrollbar
					classLayout="flex flex-col gap-1 mr-4"
					className="chat-scrollbar h-0 w-full grow px-2 pt-2 mt-3 mr-5"
				>
					{#each chatMessages as message, i}
						<ChatMessage
							on:scrollTop={() => handleTop()}
							msg={message}
							time={i === 0 || (message.time > 0 && message.time < 100)
								? message.time
								: ""}
						/>
					{/each}
				</Scrollbar>
				<!-- Loading text -->
				{#if loading}
					<LoadingAnimation />
				{/if}
			</div>
		</div>
		<!-- right Hint -->
		<LantencyHint />
	</div>
</div>



<style>
	.row::-webkit-scrollbar {
		display: none;
	}

	.row {
		scrollbar-width: none;
	}

	.row {
		-ms-overflow-style: none;
	}
</style>
