<script lang="ts">
	import improve from "$lib/assets/104b-add-to-favorites-outlined.svg";
	import clear from "$lib/assets/clear.svg";
	import transport from "$lib/assets/transport.svg";
	import financial from "$lib/assets/financial.svg";
	import stock from "$lib/assets/stock.svg";
	import education from "$lib/assets/education.svg";
	import realEstate from "$lib/assets/real_estate.svg";
	import favicon from "$lib/assets/favicon.png";
	import ChatMessage from "$lib/components/ChatMessage.svelte";
	import ScrollOptions from "$lib/components/ScrollOptions.svelte";
	import chatResponse from "./+page";

	interface ChatCompletionRequestMessage {
		role: string;
		content: string;
	}
	interface Option {
		label: string;
		value: number;
	}

	interface ModelOptions {
		[key: string]: Option[];
	}

	let is_base: boolean = true;
	let query: string = "";
	let answer: string = "";
	let loading: boolean = false;
	let chatMessages: ChatCompletionRequestMessage[] = [];
	let scrollToDiv: HTMLDivElement;
	let initMessage = `Welcome to Neural Chat! 😊`;
	let initModel: boolean = true;
	let current_model: String = "";
	let model_list = [];
	let initPage: Boolean = true;
	let optionType: string = "knowledge base";
	let outputDataInterval: NodeJS.Timeout | undefined;
	let dataQueue: string[] = [];
	let finalAnswer = "";
	let shouldExitSubmit = false;
	let is_done = false;

	const defaultTip = [
		{
			name: "What are the benefits of regular exercise?",
			icon: improve,
		},
		{ name: "How does climate change impact the environment?", icon: improve },
		{
			name: "Please provide me with a hearty lunch menu.",
			icon: improve,
		},
	];
	const tipsDict = {
		Wikipedia: [
			{ name: "What is the largest ocean in the world?", icon: improve },
			{ name: "Who was the first person to walk on the moon?", icon: improve },
			{ name: "What is the tallest mountain in the world?", icon: improve },
		],
		"INC Document": [
			{ name: "What is the quantization?", icon: improve },
			{ name: "What is the Smooth Quant?", icon: improve },
			{ name: "What is Neural Architecture Search?", icon: improve },
		],
		"llama-7b": [
			{ name: "If you could have any superpower, what would it be?", icon: improve, },
			{ name: "Can you tell me a joke?", icon: improve },
			{ name: "Can you recommend a good restaurant in Shanghai?", icon: improve, },
		],
		"gpt-j-6b": [
			{ name: "How do you define happiness?", icon: improve },
			{ name: "What are your capabilities?", icon: improve },
			{ name: "If you could travel anywhere in the world, where would you go and why?", icon: improve, },
		],
	};

	let modelOptions: ModelOptions = {
		"llama-7b": [
			{
				label: "Max Tokens",
				value: 512,
				minRange: 0,
				maxRange: 1024,
				step: 1,
			},
			{
				label: "Temperature",
				value: 0.1,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top P",
				value: 0.75,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top K",
				value: 1,
				minRange: 0,
				maxRange: 200,
				step: 1,
			},
		],
		"gpt-j-6b": [
			{
				label: "Max Tokens",
				value: 512,
				minRange: 0,
				maxRange: 1024,
				step: 1,
			},
			{
				label: "Temperature",
				value: 0.1,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top P",
				value: 0.75,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top K",
				value: 1,
				minRange: 0,
				maxRange: 200,
				step: 1,
			},
		],
	};

	let knowledgeOptions = {
		Wikipedia: [
			{
				label: "Max Tokens",
				value: 512,
				minRange: 0,
				maxRange: 1024,
				step: 1,
			},
			{
				label: "Temperature",
				value: 0.1,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top P",
				value: 0.75,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top K",
				value: 1,
				minRange: 0,
				maxRange: 200,
				step: 1,
			},
		],
		"INC Document": [
			{
				label: "Max Tokens",
				value: 512,
				minRange: 0,
				maxRange: 1024,
				step: 1,
			},
			{
				label: "Temperature",
				value: 0.1,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top P",
				value: 0.75,
				minRange: 0,
				maxRange: 1.0,
				step: 0.1,
			},
			{
				label: "Top K",
				value: 1,
				minRange: 0,
				maxRange: 200,
				step: 1,
			},
		],
	};

	let domain_list = [
		{
			title: "Stock",
			style: "border-white",
			svg: stock,
			imgPosition: "stock-position",
		},
		{
			title: "Education",
			style: "border-white",
			svg: education,
			imgPosition: "education-position",
		},
		{
			title: "Real Estate",
			style: "border-indigo-800",
			svg: realEstate,
			imgPosition: "real-position",
		},
	];

	let selected = {
		Model: Object.keys(modelOptions)[0],
		"knowledge base": Object.keys(knowledgeOptions)[0],
	};

	$: {
		if (initModel) {
			initModel = false;
			chatResponse.modelList().then(function (response) {
				model_list = response;

				function removeSuffix(str: string, suffix: string): string {
					if (str.endsWith(suffix)) {
						return str.slice(0, -suffix.length);
					}
					return str;
				}
				const suffixToRemove = "-hf-conv";

				model_list.forEach((key: any) => {
					if (typeof key === "string") {
						const result = removeSuffix(key, suffixToRemove);

						modelOptions[result] = [
							{
								label: "Max Tokens",
								value: 512,
								minRange: 0,
								maxRange: 1024,
								step: 1,
							},
							{
								label: "Temperature",
								value: 0.1,
								minRange: 0,
								maxRange: 1.0,
								step: 0.1,
							},
							{
								label: "Top P",
								value: 0.75,
								minRange: 0,
								maxRange: 1.0,
								step: 0.1,
							},
							{
								label: "Top K",
								value: 1,
								minRange: 0,
								maxRange: 200,
								step: 1,
							},
						];
					}
				});

				current_model = model_list[0];
			});
		}
	}
	$: inputLength = query.length;

	function scrollToBottom() {
		setTimeout(function () {
			scrollToDiv.scrollIntoView({
				behavior: "smooth",
				block: "end",
				inline: "nearest",
			});
		}, 100);
	}

	function outputDataFromQueue() {
		finalAnswer = answer;
		console.log(
			"finalAnswer",
			finalAnswer,
			finalAnswer.endsWith("[DONE]"),
			dataQueue
		);
		if (dataQueue.length > 0) {
			var content = dataQueue.shift();
			if (content) {
				if (content != "[DONE]") {
					if (content.startsWith(answer)) {
						console.log("now answer???", answer);
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
			answer = "";
			return;
		}
	}

	const handleSubmit = async () => {
		loading = true;
		chatMessages = [...chatMessages, { role: "Human", content: query }];
		answer = "";
		let type: any = {};
		let content = "";
		let gptOptionList = modelOptions["gpt-j-6b"];
		let llmaOptionList = modelOptions["llama-7b"];
		if (is_base) {
			type = {
				model: "llma",
				knowledge: "General",
			};
		} else {
			if (optionType == "Model") {
				if (selected.Model == "gpt-j-6b") {
					type = {
						model: "gpt-j-6b",
						temperature: gptOptionList.find(
							(option) => option.label === "Temperature"
						)?.value,
						max_new_tokens: gptOptionList.find(
							(option) => option.label === "Max Tokens"
						)?.value,
						topk: gptOptionList.find((option) => option.label === "Top K")
							?.value,
					};
				} else if (selected.Model == "llama-7b") {
					type = {
						model: "llma",
						temperature: llmaOptionList.find(
							(option) => option.label === "Temperature"
						)?.value,
						max_new_tokens: llmaOptionList.find(
							(option) => option.label === "Max Tokens"
						)?.value,
						topk: llmaOptionList.find((option) => option.label === "Top K")
							?.value,
					};
				}
			} else if (optionType == "knowledge base") {
				if (selected["knowledge base"] == "Wikipedia") {
					type = {
						model: "knowledge",
						knowledge: "WIKI",
					};
				} else if (selected["knowledge base"] == "INC Document") {
					type = {
						model: "knowledge",
						knowledge: "INC",
					};
				}
			}
		}
		const eventSource = await chatResponse.chatMessage(chatMessages, type);

		query = "";
		eventSource.addEventListener("error", handleError);
		eventSource.addEventListener("message", (e) => {
			console.log("e.data??", e, typeof e);
			scrollToBottom();
			let currentMsg = e.data;
			try {
				loading = false;
				if (dataQueue.length === 0 && currentMsg === "[DONE]") {
					console.log("already done");
					shouldExitSubmit = true; 
					
				} else if (currentMsg.startsWith("b'")) {
					console.log("coming");
					let text = currentMsg.slice(2, -1);
					const regex = /.*Assistant:((?:(?!",).)*)",/;
					const match = text.match(regex);
					content = match ? match[1].trim() : "";
					content = content
						.replace(/\\\\n/g, "")
						.replace(/\\n/g, "")
						.replace(/\n/g, "")
						.replace(/\\'/g, "'");
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
		scrollToBottom();
	};

	function handleError<T>(err: T) {
		loading = false;
		query = "";
		answer = "";
	}
</script>

<svelte:head>
	<title>Neural Chat</title>
	<meta name="description" content="Neural Chat" />
</svelte:head>

<main class="h-screen max-h-screen font-intel font-sm">
	<div class="flex h-full">
		<div class="w-2/12  px-6 pt-10 pb-8 border-r-2 bg-left">
			<div
				class="font-title-intel text-xl text-blue-300 mb-4 pb-4 border-b flex gap-4"
			>
				<img src={favicon} alt="" class="w-7 h-7" />
				{#if is_base}
					<span>Used For?</span>
				{:else}
					<span>Q&A</span>
				{/if}
			</div>
			<div class="h-5/6 mb-13 carousel carousel-vertical">
				{#if is_base}
					<div class="grid gap-6 grid-cols-1 text-title px-2 mt-4">
						<div class="btn-group flex flex-col gap-6">
							<figure
								class="w-5 group-hover:scale-110 ml-5  base-checked-figure general-img"
							>
								<img src={transport} alt="" />
							</figure>
							<input
								type="radio"
								name="options"
								data-title="General "
								class="btn py-7 my-2 general-icon"
								checked
							/>
							<figure
								class="w-5 group-hover:scale-110 ml-5 base-checked-figure financial-img"
							>
								<img src={financial} alt="" />
							</figure>
							<input
								type="radio"
								name="options"
								data-title="Financial"
								class="btn py-7 my-2"
							/>
						</div>
						{#each domain_list as domain}
							<div
								class="h-14 rounded-md btn-bg {domain.style} flex items-center gap-4 justify-left group cursor-pointer"
							>
								<figure class="w-5 group-hover:scale-110 ml-3">
									<img src={domain.svg} alt="" />
								</figure>
								<p class="text-white pl-4 font-semibold">{domain.title}</p>
							</div>
						{/each}
					</div>
				{:else}
					<div class="tabs mb-6 ">
						<button
							class="tab w-1/3 text-sm h-14 rounded-md {optionType == 'Model'
								? 'btn-bg text-blue-400'
								: 'bg-slate-950 text-gray-400 hover:bg-slate-700'}"
							class:tab-active={optionType == "Model"}
							on:click={() => {
								optionType = "Model";
								chatMessages = [];
							}}>Model</button
						>
						<button
							class="tab w-2/3 text-sm h-14 rounded-md {optionType ==
							'knowledge base'
								? 'btn-bg text-blue-400'
								: 'bg-slate-950 text-gray-400 hover:bg-slate-700'}"
							class:tab-active={optionType == "knowledge base"}
							on:click={() => {
								optionType = "knowledge base";
								chatMessages = [];
							}}>Knowledge Base</button
						>
					</div>
					<div class="flex flex-col min-h-0 grow gap-5">
						{#if optionType == "Model"}
							<ScrollOptions
								title={selected["Model"] || "Select Model"}
								lists={modelOptions}
								bind:selected={selected["Model"]}
							/>
						{:else if optionType == "knowledge base"}
							<ScrollOptions
								title={selected["knowledge base"] || "Select knowledge base"}
								lists={knowledgeOptions}
								bind:selected={selected["knowledge base"]}
							/>
						{/if}
					</div>
				{/if}
			</div>
			<div class="tabs ">
				<button
					class="tab tab-bordered w-1/2 {is_base
						? 'bg-blue-500 text-white'
						: 'bg-slate-950 text-gray-400 hover:bg-slate-700'}"
					class:tab-active={is_base}
					on:click={() => {
						is_base = true;
						chatMessages = [];
					}}>Basic</button
				>
				<button
					class="tab tab-bordered w-1/2 {!is_base
						? 'bg-blue-500 text-white'
						: 'bg-slate-950 text-gray-400 hover:bg-slate-700'}"
					class:tab-active={!is_base}
					on:click={() => {
						is_base = false;
						chatMessages = [];
					}}>Advanced</button
				>
			</div>
		</div>
		<div class="w-10/12 flex flex-col">
			<header class="flex justify-between w-full border-b-2">
				<div class="basis-1/3 flex items-center justify-center" />
				<div class="basis-1/3" />
			</header>
			<div class="flex flex-col flex-grow w-full bg-white  px-20 pt-10 mb-4">
				{#if initPage}
					<div class="flex flex-col-reverse items-start sm:flex-row">
						<div class="flex flex-col pr-8 mb-4">
							<h1
								class="mb-3 text-3xl font-bold tracking-tight text-black dark:text-white md:text-5xl"
							>
								Welcome

								<span class="relative inline-block ml-2">
									<span class="relative text-blue-600 skew-y-3"
										>Neural Chat</span
									>
								</span>
								!
							</h1>
							<p class="mb-16 text-gray-600 dark:text-gray-400">
								<a>Your AI-powered copilot for the web 😊</a>
							</p>
						</div>
						<div
							class="w-[80px] h-[80px] rounded-full sm:w-[176px] sm:h-[136px] relative mb-8 sm:mb-0 mr-auto bg-cyan-300 bg-opacity-25"
						>
							<img src="" alt="" />
						</div>
					</div>

					<section class="w-full mb-16">
						<h3
							class="mb-6 text-2xl font-bold tracking-tight text-black dark:text-white md:text-4xl"
						>
							Hints
						</h3>
						<div class="flex flex-col gap-6 md:flex-row">
							{#each defaultTip as tip}
								<div
									class="w-full transform rounded-xl bg-gradient-to-r from-sky-400 via-blue-500 to-purple-500 p-1 transition-all hover:scale-[1.01] md:w-1/3"
									on:click={() => {
										query = tip.name;
										initPage = false;
										handleSubmit();
									}}
								>
									<div
										class="flex h-full flex-col justify-between rounded-lg bg-white p-4 dark:bg-gray-900"
									>
										<div class="flex flex-col justify-between md:flex-row">
											<h4
												class="mb-6 w-full text-lg font-medium tracking-tight text-gray-900 dark:text-gray-100 sm:mb-10 md:text-lg"
											>
												{tip.name}
											</h4>
										</div>
									</div>
								</div>
							{/each}
						</div>
					</section>
				{:else}
					<div
						class="flex flex-col flex-grow h-0 p-4 overflow-auto carousel carousel-vertical"
					>
						<ChatMessage
							type="Assistant"
							message={initMessage}
							displayTimer={false}
						/>
						{#each chatMessages as message}
							<ChatMessage type={message.role} message={message.content} />
						{/each}
						{#if answer}
							<ChatMessage type="Assistant" message={answer} />
						{/if}
						{#if loading}
							<ChatMessage type="Assistant" message="Loading.." />
						{/if}
						<div class="" bind:this={scrollToDiv} />
					</div>
				{/if}
			</div>
			<footer class="w-10/12 mx-auto pb-5">
				<div class="flex justify-between items-end mb-2 gap-2 text-sm">
					{#each initPage && is_base
						? []
						: !is_base
						? optionType === "knowledge base"
						  ? tipsDict[selected["knowledge base"]]
						  : tipsDict[selected["Model"]]
						: defaultTip as tip}
						<button
							class="flex bg-title px-2 py-1 gap-2 group cursor-pointer"
							disabled={loading}
							on:click={() => {
								query = tip.name;
								initPage = false;
								handleSubmit();
							}}
						>
							<img src={tip.icon} alt="" class="w-4 opacity-50" />
							<span class="opacity-50 group-hover:opacity-90 text-left">{tip.name}</span>
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
				<textarea
					class="textarea textarea-bordered h-12 w-full"
					disabled={loading}
					placeholder="Type here..."
					maxlength="120"
					bind:value={query}
					on:keydown={(event) => {
						if (event.key === "Enter" && !event.shiftKey && query) {
							initPage = false;
							event.preventDefault();
							handleSubmit();
						}
					}}
				/>

				<div class="flex flex-row-reverse"><span>{inputLength}/120</span></div>
			</footer>
		</div>
	</div>
</main>

<style lang="postcss">
	.bg-left {
		background: rgb(36 41 51);
	}

	.btn-bg {
		background: rgb(42 48 60);
	}

	.unavailable-items {
		filter: grayscale(1);
		border-color: gray;
		color: gray;
	}

	input[type="number"]::-webkit-inner-spin-button,
	input[type="number"]::-webkit-outer-spin-button {
		opacity: 1;
	}

	.btn-group {
		display: flex;
		flex-direction: column;
		gap: 6px;
		position: relative;

		& .btn {
			border-radius: 8px;
			background: rgb(42 48 60);
			border: rgb(42 48 60);
		}
	}

	.btn {
		position: relative;
		padding-left: 30px;
		border-radius: 0;
	}

	.base-checked-figure {
		position: absolute;
		transform: translateY(-50%);
		z-index: 1;
	}

	.general-position {
		margin-top: 20%;
	}

	.stock-position,
	.financial-position {
		margin-top: 61%;
	}

	.general-img,
	.financial-img {
		position: absolute;
		z-index: 2;
	}

	.general-img {
		margin-top: 1.9rem;
	}

	.financial-img {
		margin-top: 6.9rem;
	}
</style>
