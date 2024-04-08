<script lang="ts">
	export let data;
	import { ifStoreMsg, knowledge1 } from "$lib/shared/stores/common/Store";
	import { onMount } from "svelte";
	import {
		LOCAL_STORAGE_KEY,
		MessageRole,
		MessageType,
		type Message,
	} from "$lib/shared/constant/Interface";
	import {
		fromTimeStampToTime,
		getCurrentTimeStamp,
		scrollToBottom,
		scrollToTop,
	} from "$lib/shared/Utils";
	import { fetchTextStream } from "$lib/network/chat/Network";
	import LoadingAnimation from "$lib/shared/components/loading/Loading.svelte";
	import { browser } from "$app/environment";
	import "driver.js/dist/driver.css";
	import "$lib/assets/layout/css/driver.css";
	import UploadFile from "$lib/shared/components/upload/uploadFile.svelte";
	import PaperAirplane from "$lib/assets/chat/svelte/PaperAirplane.svelte";
	import Gallery from "$lib/shared/components/chat/gallery.svelte";
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import ChatMessage from "$lib/modules/chat/ChatMessage.svelte";

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

	function isEmptyObject(obj: any): boolean {
		for (let key in obj) {
			if (obj.hasOwnProperty(key)) {
				return false;
			}
		}
		return true;
	}
</script>

<!-- <DropZone on:drop={handleImageSubmit}> -->
<div
	class="h-full items-center gap-5 bg-white sm:flex sm:pb-2 lg:rounded-tl-3xl"
>
	<div class="mx-auto flex h-full w-full flex-col sm:mt-0 sm:w-[72%]">
		<div
			class="fixed relative flex w-full flex-col items-center justify-between bg-white p-2 pb-0"
		>
			<div class="relative my-4 flex w-full flex-row justify-center">
				<div class="focus:border-none relative w-full">
					<input
						class="text-md block w-full border-0 border-b-2 border-gray-300 px-1 py-4
						text-gray-900 focus:border-gray-300 focus:ring-0 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500"
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
						><svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							width="24"
							height="24"
							class="fill-[#0597ff] group-hover:fill-[#0597ff]"
							><path
								d="M12.6 12 10 9.4 7.4 12 6 10.6 8.6 8 6 5.4 7.4 4 10 6.6 12.6 4 14 5.4 11.4 8l2.6 2.6zm7.4 8V2q0-.824-.587-1.412A1.93 1.93 0 0 0 18 0H2Q1.176 0 .588.588A1.93 1.93 0 0 0 0 2v12q0 .825.588 1.412Q1.175 16 2 16h14zm-3.15-6H2V2h16v13.125z"
							/></svg
						><span class="font-medium text-[#0597ff]">CLEAR</span></button
					>
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
		<!-- gallery -->
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
