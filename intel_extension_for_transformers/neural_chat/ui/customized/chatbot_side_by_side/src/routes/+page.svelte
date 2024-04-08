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
	import { hiddenDrawer } from "$lib/shared/stores/common/Store";
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
	import { Footer, FooterBrand, FooterCopyright } from "flowbite-svelte";

	let query: string = "";
	let loading: boolean = false;
	let scrollToDiv1: HTMLDivElement;
	let scrollToDiv2: HTMLDivElement;
	// ·········
	let chatMessages1: Message[] = data.chatMsg1 ? data.chatMsg1 : [];
	let chatMessages2: Message[] = data.chatMsg2 ? data.chatMsg2 : [];
	let displayMsg = false;

	// ··············

	$: knowledge_1 = $knowledge1?.id ? $knowledge1.id : "default";
	$: knowledge_2 = "default";

	onMount(async () => {
		scrollToDiv1 = document
			.querySelector(".chat-scrollbar1")
			?.querySelector(".svlr-viewport")!;
		scrollToDiv2 = document
			.querySelector(".chat-scrollbar2")
			?.querySelector(".svlr-viewport")!;
	});

	function handleTop(scrollToDiv) {
		console.log("scrollToDiv", scrollToDiv);

		scrollToTop(scrollToDiv);
	}

	function storeMessages(key, chatMessagesMap) {
		localStorage.setItem(key, JSON.stringify(chatMessagesMap));
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

	const callTextStream = async (query: string, group, chatMessages) => {
		let eventSource;
		if (group == "1") {
			eventSource = await fetchTextStream(query, knowledge_1, group);
		} else if (group == "2") {
			eventSource = await fetchTextStream(query, knowledge_2, group);
		}

		console.log('eventSource', eventSource);

		eventSource.addEventListener("message", (e: any) => {
			console.log('e.data', e.data);

			let currentMsg = extractContent(e.data);
			console.log('currentMsg', currentMsg);
			
			currentMsg = currentMsg.replace("@#$", " ");
			console.log("currentMsg", currentMsg);
			if (currentMsg == "[DONE]") {
				console.log("done getCurrentTimeStamp", getCurrentTimeStamp);
				let startTime = chatMessages[chatMessages.length - 1].time;

				loading = false;
				let totalTime = parseFloat(
					((getCurrentTimeStamp() - startTime) / 1000).toFixed(2)
				);

				if (chatMessages.length - 1 !== -1) {
					chatMessages[chatMessages.length - 1].time = totalTime;
				}
				if (group == "1") {
					chatMessages1 = [...chatMessages];
				} else if (group == "2") {
					chatMessages2 = [...chatMessages];
				}

				if (group == "1") {
					storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY1, chatMessages);
				} else if (group == "2") {
					storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2, chatMessages);
				}
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
					if (group == "1") {
						chatMessages1 = [...chatMessages];
					} else if (group == "2") {
						chatMessages2 = [...chatMessages];
					}
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

					if (group == "1") {
						chatMessages1 = [...chatMessages];
					} else if (group == "2") {
						chatMessages2 = [...chatMessages];
					}
				}
				scrollToBottom(scrollToDiv1);
				scrollToBottom(scrollToDiv2);
			}
		});
		eventSource.stream();
	};

	const handleTextSubmit = async (group, chatMessages) => {
		loading = true;
		const newMessage = {
			role: MessageRole.User,
			type: MessageType.Text,
			content: query,
			time: 0,
			first_token_latency: "0",
			msecond_per_token: "0",
		};
		chatMessages = [...chatMessages, newMessage];
		console.log("group", group);

		if (group == "1") {
			chatMessages1 = [...chatMessages];
		} else if (group == "2") {
			chatMessages2 = [...chatMessages];
		}
		chatMessages2 = [...chatMessages2];
		displayMsg = true;
		scrollToBottom(scrollToDiv1);
		scrollToBottom(scrollToDiv2);
		if (group == "1") {
			storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY1, chatMessages);
		} else if (group == "2") {
			storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2, chatMessages);
		}

		await callTextStream(newMessage.content, group, chatMessages);

		scrollToBottom(scrollToDiv1);
		scrollToBottom(scrollToDiv2);

		if (group == "1") {
			storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY1, chatMessages);
		} else if (group == "2") {
			storeMessages(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2, chatMessages);
		}
	};

	function handelClearHistory(group) {
		if (group == "1") {
			localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY1);
			chatMessages1 = [];
		} else if (group == "2") {
			localStorage.removeItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2);
			chatMessages2 = [];
		}
	}

	function isEmptyObject(obj: any): boolean {
		for (let key in obj) {
			if (obj.hasOwnProperty(key)) {
				return false;
			}
		}
		return true;
	}

	async function handleSubmit() {
		await Promise.all([
			handleTextSubmit("1", chatMessages1),
			handleTextSubmit("2", chatMessages2),
		]);
		query = "";
		scrollToBottom(scrollToDiv1);
		scrollToBottom(scrollToDiv2);
	}
</script>

<!-- <DropZone on:drop={handleImageSubmit}> -->
<div
	class="h-full items-center gap-5 bg-[#183544] bg-fixed sm:flex lg:rounded-tl-3xl"
	style="background-image: url('/src/lib/assets/png/Picture2.png'); background-size: cover; background-position: center;"
>
	<!-- <div class="mx-auto flex h-full w-full flex-col sm:mt-0 sm:w-[72%]"> -->
	<div
		class={`${
			$hiddenDrawer ? "mx-auto" : "mx-3"
		} flex h-full w-full flex-col sm:mt-0 sm:w-[72%]`}
	>
		<div class="flex justify-between p-2">
			<p class="text-[1.7rem] font-bold tracking-tight text-white">ChatBot</p>
			<!-- <UploadFile /> -->
		</div>
		<div
			class="fixed relative flex w-full flex-col items-center justify-between bg-[#183544] p-2 pb-0"
		>
			<div class="relative my-4 flex w-full flex-row justify-center">
				<div class="focus:border-none relative w-full">
					<input
						class="text-md block w-full border-0 border-b-2 border-gray-300 bg-[#183544] px-1 py-4
						text-white placeholder-white focus:border-gray-300 focus:ring-0 dark:border-gray-600 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400 dark:focus:border-blue-500 dark:focus:ring-blue-500"
						type="text"
						placeholder="Enter prompt here"
						disabled={loading}
						maxlength="1200"
						bind:value={query}
						on:keydown={(event) => {
							if (event.key === "Enter" && !event.shiftKey && query) {
								event.preventDefault();
								handleSubmit();
							}
						}}
					/>
					<button
						on:click={() => {
							if (query) {
								handleSubmit();
							}
						}}
						type="submit"
						class="absolute bottom-2.5 end-2.5 px-4 py-2 text-sm font-medium text-white dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800"
						><PaperAirplane /></button
					>
				</div>
			</div>
		</div>

		<!-- side by side UI -->

		<div class="grid h-full grid-cols-2 gap-4  sm:gap-8">
			<div
				class="dark:hover:shadow-lg-light flex w-full !max-w-none max-w-sm flex-col divide-gray-200 rounded-lg border border-gray-200 bg-[#183544] text-gray-500 shadow-none hover:bg-gray-100 hover:shadow-lg dark:divide-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-900"
			>
				<div
					class="flex items-center justify-center rounded-t-md border-b border-gray-200 bg-indigo-700 px-5 py-2.5 text-center dark:border-gray-700 dark:bg-gray-700"
				>
					<span
						class="text-center text-base font-medium text-white dark:text-white"
						>SPR</span
					>
				</div>
				<div class=" flex h-full items-center justify-center bg-[#0f2631]">
					{#if !isEmptyObject(chatMessages2) || displayMsg}
						<div class="flex h-full w-full flex-col rounded border p-2 shadow">
							<!-- gallery -->
							<Gallery
								chatMessages={chatMessages1}
								scrollName={"chat-scrollbar1"}
								label={"Gaudi2"}
								on:ExternalClear={() => handelClearHistory("1")}
								on:ExternalTop={() => handleTop(scrollToDiv1)}
							/>
						</div>
					{/if}
				</div>
			</div>
			<div
				class="dark:hover:shadow-lg-light flex w-full !max-w-none max-w-sm flex-col divide-gray-200 rounded-lg border border-gray-200 bg-[#183544] text-gray-500 shadow-none hover:bg-gray-100 hover:shadow-lg dark:divide-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-900"
			>
				<div
					class="flex items-center justify-center rounded-t-md border-b border-sky-200 bg-sky-300 px-5 py-2.5 text-center dark:border-gray-700 dark:bg-gray-700"
				>
					<span
						class="text-center text-base font-medium text-gray-900 dark:text-white"
						>Latest Xeon Gen</span
					>
				</div>
				<div class="flex h-full items-center justify-center bg-[#0f2631]">
					{#if !isEmptyObject(chatMessages2) || displayMsg}
						<div class="flex h-full w-full flex-col rounded border p-2 shadow">
							<!-- gallery -->
							<Gallery
								chatMessages={chatMessages2}
								scrollName={"chat-scrollbar2"}
								label={"Gaudi2"}
								on:ExternalClear={() => handelClearHistory("2")}
								on:ExternalTop={() => handleTop(scrollToDiv2)}
							/>
						</div>
					{/if}
				</div>
			</div>
		</div>
		{#if loading}
			<LoadingAnimation />
		{/if}
		<!-- side by side UI -->
		<!-- <div class="flex justify-center py-2">
			<FooterBrand
				imgClass="w-16 h-16 m-1"
				src="/src/lib/assets/png/intelXeon.png"
				alt="Flowbite Logo"
				name="Flowbite"
			/>
			<FooterBrand
				imgClass="w-16 h-16 m-1"
				src="/src/lib/assets/png/intelGaudi.png"
				alt="Flowbite Logo"
				name="Flowbite"
			/>
		</div> -->
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
