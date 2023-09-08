<script lang="ts">
	import ChatMessage from "$lib/modules/chat/chat-message.svelte";

	// icon
	import LoadingButtonSpinnerIcon from "$lib/assets/icons/loading-button-spinner-icon.svelte";
	import ArrowPathIcon from "$lib/assets/icons/arrow-path-icon.svelte";

	// tool
	import { MESSAGE_ROLE, type Message } from "$lib/components/shared/shared.type";
	import { TalkingTemplateLibrary, TalkingVoiceLibrary } from '$lib/components/talkbot/constant';
	import { scrollToBottom } from "$lib/components/shared/shared-utils";
	import chatResponse from "$lib/modules/chat/network";
	import VoiceButton from "../talkbot/voice-button.svelte";
	import { CollectionType, currentVoice, currentKnowledge, TalkingKnowledgeCustom, TalkingVoiceCustom } from "../talkbot/store";
	
	let loading: boolean = false;
	let scrollToDiv: HTMLDivElement;

	let chatMessages: Message[] = [];

	$: enableRegenerateMessage = !loading && chatMessages.length > 2;

	const handleSubmit = async (enableRegenerate = false): Promise<void> => {
		scrollToBottom(scrollToDiv);
		loading = true;
		if (enableRegenerate) {
			let lastRole = chatMessages[chatMessages.length - 1];
			if (lastRole.role === MESSAGE_ROLE.ASSISTANT) {
				chatMessages = chatMessages.filter(
					(_, i: number) => i !== chatMessages.length - 1
				);
			}
		}

		const content = chatMessages[chatMessages.length - 1].content as string;
		
		const blob = await fetch(content).then(r => r.blob());
		const voice = ($currentVoice.collection === CollectionType.TemplateLibrary ? TalkingTemplateLibrary[$currentVoice.id].identify
						: ($currentVoice.collection === CollectionType.Custom ? $TalkingVoiceCustom[$currentVoice.id].id 
						: ($currentVoice.collection === CollectionType.Library ? TalkingVoiceLibrary[$currentVoice.id].identify : "default")
						))
		const knowledge = ($currentKnowledge.collection === CollectionType.Custom ?
						$TalkingKnowledgeCustom[$currentKnowledge.id].id : 'default')	
								
		const res = await chatResponse.fetchAudioText(blob);
		console.log('res ---', res);
		const eventSource = await chatResponse.fetchAudioStream(res.asr_result, voice, knowledge)

		eventSource.addEventListener("message", async (e) => {
			loading = false;
			let currentMsg = e.data;
			if (currentMsg.startsWith("b'")) {
				const audioUrl = "data:audio/wav;base64," + currentMsg.slice(2, -1)
				const blob = await fetch(audioUrl).then(r => r.blob());
				if (chatMessages[chatMessages.length - 1].role == MESSAGE_ROLE.USER) {
					chatMessages = [...chatMessages, { role: MESSAGE_ROLE.ASSISTANT, content: [URL.createObjectURL(blob),] }]
				} else {
					let content = chatMessages[chatMessages.length - 1].content
					chatMessages[chatMessages.length - 1].content = [...content, URL.createObjectURL(blob)]
				}
				scrollToBottom(scrollToDiv);
			} else if (currentMsg === '[DONE]') {
				let content = chatMessages[chatMessages.length - 1].content
				chatMessages[chatMessages.length - 1].content = [...content, 'done']
			}
		});

		eventSource.stream();
	};
</script>

<svelte:head>
	<title>Welcome to NeuralChat</title>
	<meta name="description" content="Neural Chat" />
</svelte:head>

<div class="flex h-full w-full flex-col p-10 py-4">
	<div
		class="carousel carousel-vertical flex h-0 flex-grow flex-col overflow-auto p-4"
		bind:this={scrollToDiv}
	>
		<ChatMessage
			type="Assistant"
			message={`Welcome to Neural Chat! 😊`}
			displayTimer={false}
		/>
		{#each chatMessages as message, idx (message)}
			<ChatMessage
				type={message.role}
				message={message.content}
			/>
		{/each}
	</div>

	<div class="mx-auto w-10/12 pb-5">
		<!-- Loading text -->
		{#if loading}
			<div
				class="mb-6 flex items-center justify-center self-center text-sm text-gray-500"
			>
				<div class="inset-y-0 left-0 pl-2">
					<LoadingButtonSpinnerIcon />
				</div>

				<div>Talking Bot is thinking...</div>
			</div>
		{/if}

		<!-- regenerate -->
		{#if enableRegenerateMessage}
			<button
				on:click={() => {
					handleSubmit((true));
				}}
				type="button"
				class="mx-auto mb-1 flex w-48 items-center justify-center gap-2 self-center whitespace-nowrap rounded-md bg-white px-3 py-2 text-sm text-gray-700 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
			>
				<ArrowPathIcon />
				Regenerate response
			</button>
		{/if}

		<!-- Input -->
		<div class="flex flex-col items-center">
			<VoiceButton bind:chatMessages on:done={() => {handleSubmit()}} />
		</div>
	</div>
</div>
