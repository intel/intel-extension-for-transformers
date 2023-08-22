<script lang="ts">
	export let chatId = "";

	// icon
	import clear from "$lib/assets/clear.svg";
	import PaperAirplane from "$lib/assets/icons/paper-airplane.svelte";
	import LoadingButtonSpinnerIcon from "$lib/assets/icons/loading-button-spinner-icon.svelte";
	import ArrowPathIcon from "$lib/assets/icons/arrow-path-icon.svelte";
	// svelte
	import ChatMessage from "$lib/modules/chat/chat-message.svelte";
	import SideBar from "$lib/modules/chat-sidebar/side-bar.svelte";
	import LeftIconNav from "$lib/modules/chat-sidebar/left-icon-nav.svelte";
	import SideBarHeader from "$lib/modules/chat-sidebar/sidebar-header.svelte";
	import BasicChoose from "$lib/modules/chat-sidebar/basic-choose.svelte";
	import AdvanceChoose from "$lib/modules/chat-sidebar/advance-choose.svelte";
	import SideBarArrow from "$lib/modules/chat-sidebar/sidebar-arrow.svelte";
	import InitPage from "$lib/modules/chat/init-page.svelte";
	// tool
	import type { Message, Chat } from "$lib/shared/shared.type";
	import {
		DEFAULT_TIP,
		TIPS_DICT,
		MODEL_OPTION,
		KNOWLEDGE_OPTIONS,
	} from "$lib/shared/shared.constant";
	import { chats$ } from "$lib/shared/shared.store";
	import {
		upsertChat,
		scrollToBottom,
		defineType,
	} from "$lib/shared/shared-utils";
	import chatResponse from "./network";

	let domain_idx: number = 0;
	let query: string = "";
	let answer: string = "";
	let finalAnswer = "";
	let dataQueue: string[] = [];
	let initMessage = `Welcome to Neural Chat! 😊`;
	let loading: boolean = false;
	let is_done: boolean = false;
	let shouldExitSubmit: boolean = false;
	let initPage: Boolean = true;
	let isBoardVisible: boolean = true;
	let isHistoryVisible: boolean = true;
	let enableRegenerate: boolean = false;
	let outputDataInterval: NodeJS.Timeout | undefined;
	let scrollToDiv: HTMLDivElement;

	const chat = chatId && $chats$?.[chatId];
	let chatMessages: Message[] = (chat as Chat)?.messages?.filter(Boolean) || [];
	let mode: string = (chat as Chat)?.mode || "basic";
	let optionType: string = (chat as Chat)?.optionType || "Model";
	let initModel =
		(chat as Chat)?.MODEL_OPTION || JSON.parse(JSON.stringify(MODEL_OPTION));
	let api_key = (chat as Chat)?.api_key || "";
	let selected = (chat as Chat)?.selected || {
		Model: initModel.names[0],
		"knowledge base": KNOWLEDGE_OPTIONS[0],
	};
	let article = "";

	$: inputLength = query.length;
	$: enableRegenerateMessage = !loading && chatMessages.length > 2;

	function insertChat() {
		upsertChat(
			chatId,
			chatMessages,
			mode,
			optionType,
			selected,
			initModel,
			article,
			api_key
		);
	}

	function outputDataFromQueue(): void {
		finalAnswer = answer;

		if (dataQueue.length > 0) {
			var content = dataQueue.shift();
			if (content) {
				if (content != "[DONE]") {
					if (content.startsWith(answer)) {
						answer = content;
					} else {
						answer = (answer ? answer + " " : "") + content;
					}
				} else {
					is_done = true;
				}
			}
		}

		if (is_done) {
			is_done = false;
			clearInterval(outputDataInterval);
			chatMessages = [
				...chatMessages,
				{ role: "Assistant", content: finalAnswer },
			];
			insertChat();
			answer = "";

			return;
		}
	}

	const handleSubmit = async (enableRegenerate: boolean): Promise<void> => {
		answer = "";
		let type: any = {};
		let content = "";
		let articles = [];
		loading = true;
		if (enableRegenerate) {
			let lastRole = chatMessages[chatMessages.length - 1];
			if (lastRole?.role === "Assistant") {
				chatMessages = chatMessages.filter(
					(_, i: number) => i !== chatMessages.length - 1
				);
			}
		} else {
			chatMessages = [...chatMessages, { role: "Human", content: query }];
		}

		insertChat();

		if (mode === "basic") {
			type = {
				model: "llma",
				knowledge: "General",
			};
		} else if (mode === "advanced") {
			[type, articles] = defineType(optionType, selected, initModel, article);
		}

		const eventSource = chatResponse.chatMessage(chatMessages, type, articles);
		query = "";
		eventSource.addEventListener("error", handleError);
		eventSource.addEventListener("message", (e) => {
			let currentMsg = e.data;
			try {
				loading = false;
				if (dataQueue.length === 0 && currentMsg === "[DONE]") {
					shouldExitSubmit = true;
				} else if (currentMsg.startsWith("b'")) {
					content = chatResponse.regFunc(currentMsg);
				} else {
					content = currentMsg;
				}
				dataQueue.push(content);
			} catch (err) {
				handleError(err);
			}
		});
		outputDataInterval = setInterval(outputDataFromQueue, 100);

		eventSource.stream();
		scrollToBottom(scrollToDiv);
	};

	const handleChatGPT = async (enableRegenerate: boolean) => {
		loading = true;
		if (enableRegenerate) {
			let lastRole = chatMessages[chatMessages.length - 1];
			if (lastRole?.role === "assistant") {
				chatMessages = chatMessages.filter(
					(_, i: number) => i !== chatMessages.length - 1
				);
			}
		} else {
			chatMessages = [...chatMessages, { role: "user", content: query }];
		}
		insertChat();
		answer = "";
		const eventSource = chatResponse.chatGPT(chatMessages, api_key);

		query = "";
		eventSource.addEventListener("error", handleError);
		eventSource.addEventListener("message", (e) => {
			try {
				let resp = JSON.parse(e.source.xhr.response);
				answer = resp["choices"][0]["message"]["content"];
				clearInterval(outputDataInterval);
				chatMessages = [
					...chatMessages,
					{ role: "assistant", content: answer },
				];
				insertChat();
				answer = "";
				loading = false;
				is_done = false;
			} catch (error) {
				handleError(error);
			}
		});
		eventSource.stream();
		scrollToBottom(scrollToDiv);
	};

	function handleError<T>(err: T) {
		loading = false;
		query = "";
		answer = "";
		console.log("err --->", err);
	}
</script>

<svelte:head>
	<title>Neural Chat</title>
	<meta name="description" content="Neural Chat" />
</svelte:head>

<div class="w-12">
	<LeftIconNav bind:mode bind:chatMessages />
</div>
<div class="flex grow">
	<div
		class="w-2/12 relative px-6 pt-10 pb-8 border-r-2 bg-gray-800"
		class:w-0={!isBoardVisible}
		class:px-0={!isBoardVisible}
	>
		<SideBarHeader bind:mode />

		<div class="h-5/6 mb-13 carousel carousel-vertical">
			{#if mode == "basic"}
				<BasicChoose bind:domain_idx />
			{:else if mode == "advanced"}
				<AdvanceChoose
					bind:optionType
					bind:selected
					bind:article
					MODEL_OPTION={initModel}
					{KNOWLEDGE_OPTIONS}
				/>
			{:else if mode == "chatgpt"}
				<div class="py-2 text-white">OpenAI API Key</div>
				<input
					type="text"
					placeholder="Type here..."
					bind:value={api_key}
					class="input input-bordered w-full max-w-xs"
				/>
			{/if}
		</div>
		<SideBarArrow bind:isBoardVisible bind:isHistoryVisible label="leftArrow" />
	</div>
	<div
		class="flex flex-col {`w-${
			12 - (Number(isBoardVisible) + Number(isHistoryVisible)) * 2
		}/12`}"
	>
		<div class="flex flex-col flex-grow bg-white px-16 pt-10 mb-4">
			{#if initPage && chatMessages.length < 1}
				<InitPage
					{mode}
					on:tipclick={(e) => {
						query = e.detail;
						initPage = false;
						handleSubmit(enableRegenerate = false);
					}}
				/>
			{:else}
				<div
					class="flex flex-col flex-grow h-0 p-4 overflow-auto carousel carousel-vertical"
				>
					<ChatMessage
						type="system"
						message={initMessage}
						displayTimer={false}
					/>
					{#each chatMessages as message}
						<ChatMessage type={message.role} message={message.content} />
					{/each}
					{#if answer}
						<ChatMessage type="Assistant" message={answer} />
					{/if}
					<div class="" bind:this={scrollToDiv} />
				</div>
			{/if}
		</div>
		<footer class="w-10/12 mx-auto pb-5">
			<!-- Loading text -->
			{#if loading}
				<div
					class="flex self-center items-center text-gray-500 text-sm justify-center mb-6"
				>
					<div class="inset-y-0 left-0 pl-2">
						<LoadingButtonSpinnerIcon />
					</div>

					<div>Neural Chat is thinking...</div>
				</div>
			{/if}

			<!-- regenerate -->
			{#if enableRegenerateMessage}
				<button
					on:click={() => {
						initPage = false;
						if (mode == "chatgpt") handleChatGPT((enableRegenerate = true));
						else handleSubmit((enableRegenerate = true));
					}}
					type="button"
					class="flex justify-center items-center gap-2 w-48 self-center whitespace-nowrap rounded-md mb-1 bg-white py-2 px-3 text-sm text-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 mx-auto"
				>
					<ArrowPathIcon />
					Regenerate response
				</button>
			{/if}
			<div class="flex justify-between items-end mb-2 gap-2 text-sm">
				{#each initPage && mode == "chatgpt" ? [] : mode != "basic" ? (optionType === "knowledge base" ? TIPS_DICT[selected["knowledge base"]] : TIPS_DICT[selected["Model"]]) : DEFAULT_TIP as tip}
					<button
						class="flex bg-title px-2 py-1 gap-2 group cursor-pointer"
						disabled={loading}
						on:click={() => {
							query = tip.name;
							initPage = false;
							handleSubmit(enableRegenerate = false);
						}}
					>
						<img src={tip.icon} alt="" class="w-4 opacity-50" />
						<span class="opacity-50 group-hover:opacity-90 text-left"
							>{tip.name}</span
						>
					</button>
				{/each}
				<div class="grow" />
				<button
					class="btn gap-2 bg-sky-900 hover:bg-sky-700"
					on:click={() => {
						chatMessages = [];
					}}
				>
					<img src={clear} alt="" class="w-5" />
					<span class="text-white">New Topic</span>
				</button>
			</div>

			<!-- Input -->
			<div class="flex justify-center items-center">
				<div class="relative w-full flex justify-center items-center">
					<!-- Textarea -->
					<textarea
						class="textarea textarea-bordered h-12 w-full"
						disabled={(api_key == "" && mode == "chatgpt") || loading}
						placeholder={mode == "chatgpt" && !api_key
							? "Please enter your OpenAI API key first"
							: "Type here ..."}
						maxlength="120"
						bind:value={query}
						on:keydown={(event) => {
							if (event.key === "Enter" && !event.shiftKey && query) {
								initPage = false;
								event.preventDefault();
								if (mode == "chatgpt") handleChatGPT(enableRegenerate = false);
								else handleSubmit(enableRegenerate = false);
							}
						}}
					/>

					<!-- Send button -->
					<button
						on:click={() => {
							if (query) {
								initPage = false;
								if (mode == "chatgpt") handleChatGPT(enableRegenerate = false);
								else handleSubmit(enableRegenerate = false);
							}
						}}
						type="submit"
						class="absolute right-0 inset-y-0 py-2 pr-3"
					>
						<PaperAirplane />
					</button>
				</div>
			</div>
			<div class="flex flex-row-reverse"><span>{inputLength}/120</span></div>
		</footer>
	</div>

	<div
		class="w-2/12 relative px-6 bg-gray-800"
		class:w-0={!isHistoryVisible}
		class:px-0={!isHistoryVisible}
	>
		<SideBar {MODEL_OPTION} {article} {selected} {api_key} />
		<SideBarArrow
			bind:isBoardVisible
			bind:isHistoryVisible
			label="rightArrow"
		/>
	</div>
</div>
