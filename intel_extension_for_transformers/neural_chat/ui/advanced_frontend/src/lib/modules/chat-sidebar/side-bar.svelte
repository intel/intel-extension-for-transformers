<script lang="ts">
	import { nanoid } from "nanoid";
	import Fuse from "fuse.js";
	import { goto } from "$app/navigation";
	import MagnifyingGlassIcon from "$lib/assets/icons/magnifying-glass-icon.svelte";
	import PlusIcon from "$lib/assets/icons/plus-icon.svelte";
	import { banners$, chatList$, chats$ } from "$lib/shared/shared.store";
	import {
		createNewChat,
		createNewChatListItem,
	} from "$lib/shared/shared-utils";
	import {
		BANNER_TYPE,
		ERROR,
		LOCAL_STORAGE_KEY,
		type ModelOptionType,
		type SelectedType
	} from "$lib/shared/shared.type";
	import SidebarChatItem from "./sidebar-chat-item.svelte";

	export let MODEL_OPTION: ModelOptionType
	export let article: string
	export let selected: SelectedType
	export let api_key: string

	const chatListFuseOptions = {
		// Lower threshold = closer match
		threshold: 0.3,
		keys: ["title"],
	};

	const chatsFuseOptions = {
		// Lower threshold = closer match
		threshold: 0.5,
		keys: ["messages.content"],
	};

	let searchInput;
	let isSearchInputFocused = false;
	let searchQuery = "";

	$: chatListFuse = new Fuse($chatList$, chatListFuseOptions);
	$: chatsFuse = new Fuse(Object.values($chats$), chatsFuseOptions);

	$: searchedChats = chatsFuse.search(searchQuery).map((result) => result.item);
	$: searchedChatList = chatListFuse
		.search(searchQuery)
		.map((result) => result.item);

	$: matchedChatIds = [
		...new Set([
			...searchedChats.map((chat) => chat.chatId),
			...searchedChatList.map((chat) => chat.chatId),
		]),
	];

	$: chatList = searchQuery
		? $chatList$.filter((chat) => matchedChatIds.includes(chat.chatId))
		: $chatList$;

	const handleSearchFocus = () => {
		isSearchInputFocused = true;
	};

	const handleSearchBlur = () => {
		isSearchInputFocused = false;
	};

	/**
	 * Create new chat
	 * newChatId
	 * chatList -> createNewChatListItem - share.type
	 * chats -> createNewChat - share.type -> DEFAULT_SYSTEM_MESSAGE
	 * localStorage -> LOCAL_STORAGE_KEY.CHAT_LIST、newChatId
	 */
	const handleCreateNewChat = () => {
		const newChatId = nanoid(5);

		chatList$.update((chatList) => {
			chatList.unshift(createNewChatListItem(newChatId));
			return chatList;
		});
		chats$.update((chats) => {
			chats[newChatId] = createNewChat(
				newChatId,
				[],
				"basic",
				"knowledge base",
				selected,
				MODEL_OPTION,
				article,
				api_key
			);
			return chats;
		});

		try {
			localStorage.setItem(
				LOCAL_STORAGE_KEY.CHAT_LIST,
				JSON.stringify($chatList$)
			);
			localStorage.setItem(newChatId, JSON.stringify($chats$[newChatId]));
		} catch (e: any) {
			banners$.update((banners) => {
				banners.push({
					id: ERROR.LOCAL_STORAGE_SET_ITEM,
					bannerType: BANNER_TYPE.ERROR,
					title: "Access to browser storage failed",
					description: e?.message || e?.name || "",
				});
				return banners;
			});
		}

		goto(`/chat/${newChatId}`);
	};
</script>

<!-- <svelte:window on:keydown={handleKeydown} /> -->

<div class="flex flex-1 flex-col h-full overflow-auto carousel carousel-vertical">
	<nav class="flex-1 px-1 py-3 bg-gray-800 space-y-1">
		<!-- New chat -->
		<button
			on:click={handleCreateNewChat}
			class={`w-full text-gray-300 hover:bg-gray-500 flex items-center px-2 py-3 text-sm font-medium rounded-md mb-2 shadow-sm ring-1 ring-inset ring-gray-400 `}
		>
			<PlusIcon
				overrideClasses={"text-gray-400 hover:text-gray-500 mr-3 flex-shrink-0 h-5 w-5"}
			/>
			New chat
		</button>

		<!-- Search -->
		<div class="relative flex flex-grow">
			<div
				class="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3"
			>
				<MagnifyingGlassIcon overrideClasses={`h-5 w-5 text-gray-400`} />
			</div>
			<input
				bind:value={searchQuery}
				bind:this={searchInput}
				on:focus={handleSearchFocus}
				on:blur={handleSearchBlur}
				placeholder="Search"
				type="text"
				name="search"
				class="block w-full rounded-md border-0 py-1.5 pl-10 text-gray-900 ring-1 ring-inset ring-gray-300 placeholder:text-gray-400 focus:ring-2 focus:ring-inset focus:ring-indigo-600 sm:text-sm sm:leading-6"
			/>
		</div>

		{#each chatList as { chatId: cId, title }}
			<SidebarChatItem chatId={cId} {title} />
		{/each}

		<!-- Empty state -->
		{#if !chatList.length || chatList.length === 0}
			<div class="flex flex-col items-center justify-center pt-60">
				<div class="text-gray-400 text-sm">No chats found</div>
				<button
					on:click={handleCreateNewChat}
					class="mt-2 text-gray-400 hover:text-gray-500 text-sm border border-gray-400 rounded-md px-2 py-1"
				>
					Create new chat
				</button>
			</div>
		{/if}
	</nav>
</div>
