<script lang="ts">
	import { nanoid } from "nanoid";
	import Fuse from "fuse.js";
	import PlusIcon from "$lib/assets/icons/plus-icon.svelte";
	import {
		banners$,
		chatList$,
		chats$,
	} from "$lib/components/shared/shared.store";
	import {
		createNewChat,
		createNewChatListItem,
	} from "$lib/components/shared/shared-utils";
	import {
		BANNER_TYPE,
		ERROR,
		LOCAL_STORAGE_KEY,
	} from "$lib/components/shared/shared.type";
	import SidebarChatItem from "$lib/modules/chat/sidebar-chat-item.svelte";

	export let currentChatID: string;
	
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
			chatList.unshift(createNewChatListItem(newChatId, "New Chat"));
			return chatList;
		});
		chats$.update((chats) => {
			chats[newChatId] = createNewChat(newChatId, []);
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
		currentChatID = newChatId
	};
</script>

<nav class="flex-1 space-y-1 px-1 pb-4 h-full w-full">
	<!-- Search -->
	<div class="group relative mb-2 flex h-full w-full items-center h-full">
		<div
			class="absolute block flex h-10 w-auto cursor-pointer items-center justify-center p-3 pr-2 text-sm uppercase text-gray-500 sm:hidden"
		>
			<svg
				fill="none"
				class="relative h-5 w-5"
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				stroke="currentColor"
				viewBox="0 0 24 24"
				><path
					d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
				/></svg
			>
		</div>
		<svg
			class="pointer-events-none absolute left-0 ml-4 hidden h-4 w-4 fill-current text-gray-500 group-hover:text-gray-400 sm:block"
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 20 20"
			><path
				d="M12.9 14.32a8 8 0 1 1 1.41-1.41l5.35 5.33-1.42 1.42-5.33-5.34zM8 14A6 6 0 1 0 8 2a6 6 0 0 0 0 12z"
			/></svg
		>
		<input
			bind:value={searchQuery}
			bind:this={searchInput}
			on:focus={handleSearchFocus}
			on:blur={handleSearchBlur}
			placeholder="Search"
			type="text"
			name="search"
			class="block w-full rounded-2xl bg-gray-100 py-2 pl-10 pr-4 leading-normal text-gray-400 ring-opacity-90 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500"
		/>
	</div>

	<!-- New chat -->
	<button
		on:click={handleCreateNewChat}
		class={`mb-2 flex h-10 w-full items-center rounded-md px-2 py-3 text-sm font-medium shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-[#ebf1f9] hover:text-black `}
	>
		<PlusIcon
			overrideClasses={"hover:text-gray-500 mr-3 flex-shrink-0 h-5 w-5"}
		/>
		New chat
	</button>

		<!-- Empty state -->
		{#if !chatList.length || chatList.length === 0}
		<div class="flex flex-col items-center justify-center py-6">
			<div class="text-sm">No chats found</div>
			<button
				on:click={handleCreateNewChat}
				class="mt-2 rounded-md border border-gray-400 px-2 py-1 text-sm hover:text-gray-500"
			>
				Create new chat
			</button>
		</div>
	{/if}

	<div class="carousel carousel-vertical mt-4 sm:h-[5rem] md:h-[10rem] lg:h-[20rem] xl:h-[26rem] overflow-auto">
		{#each chatList as { chatId: cId, title }}
			<SidebarChatItem chatId={cId} {title} bind:currentChatID />
		{/each}
	</div>


</nav>
