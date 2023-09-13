<script lang="ts">
  import ChatBubbleLeftIcon from '$lib/assets/icons/chat-bubble-left-icon.svelte';
  import { truncateString } from '$lib/components/shared/shared-utils';
  import { banners$, chatList$, chats$ } from '$lib/components/shared/shared.store';
  import { BANNER_TYPE, ERROR, LOCAL_STORAGE_KEY } from '$lib/components/shared/shared.type';
  import PencilSquareIcon from '$lib/assets/icons/pencil-square-icon.svelte';
  import TrashIcon from '$lib/assets/icons/trash-icon.svelte';
  import CheckIcon from '$lib/assets/icons/check-icon.svelte';
  import XMarkIcon from '$lib/assets/icons/x-mark-icon.svelte';

  export let chatId: string;
  export let title: string;
  export let currentChatID: string;

  let isEditing = false;
  let isHovering = false;
  let titleInput = title;

  /**
   * Hover
   */
  const handleMouseEnter = () => {
    isHovering = true;
  };

  const handleMouseLeave = () => {
    isHovering = false;
  };

  /**
   * Title edits
   */
  const handleTitleEditClick = () => {
    isEditing = true;
  };

  const handleCancelTitleEditClick = () => {
    isEditing = false;
  };
  const handleSaveTitleEditClick = () => {
    chatList$.update((chatList) => {
      chatList = chatList.map((chat) => {
        if (chat.chatId === chatId) {
          chat.title = titleInput;
        }
        return chat;
      });
      return chatList;
    });
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY.CHAT_LIST, JSON.stringify($chatList$));
    } catch (e: any) {
      banners$.update((banners) => {
        banners.push({
          id: ERROR.LOCAL_STORAGE_SET_ITEM,
          bannerType: BANNER_TYPE.ERROR,
          title: 'Access to browser storage failed',
          description: e?.message || e?.name || ''
        });
        return banners;
      });
    }
    isEditing = false;
  };

  /**
   * Delete chat
   */
  const handleDeleteChat = (chatId: string) => {
    chatList$.update((chatList) => {
      chatList = chatList.filter((chat) => chat.chatId !== chatId);
      return chatList;
    });
    chats$.update((chats) => {
      delete chats[chatId];
      return chats;
    });

    try {
      localStorage.setItem(LOCAL_STORAGE_KEY.CHAT_LIST, JSON.stringify($chatList$));
      localStorage.removeItem(chatId);
    } catch (e: any) {
      banners$.update((banners) => {
        banners.push({
          id: ERROR.LOCAL_STORAGE_SET_ITEM,
          bannerType: BANNER_TYPE.ERROR,
          title: 'Access to browser storage failed',
          description: e?.message || e?.name || ''
        });
        return banners;
      });
    }
    
    if (currentChatID === chatId) {
      currentChatID = ''
    }
  };

  /**
   * Select chat
   */
  const handleSelectChat = () => {
    if (isEditing) {
      return;
    }
    currentChatID = chatId
  }
</script>

<button
  on:click={handleSelectChat}
  on:mouseenter={handleMouseEnter}
  on:mouseleave={handleMouseLeave}
  type="button"
  class={`relative w-full hover:bg-[#ebf1f9]  hover:text-black group flex items-center px-2 py-3 text-sm font-medium rounded-md ${
    chatId === currentChatID ? `bg-[#1d4dd5] text-white` : ''
  }`}
>
  <!-- Title  -->
  <div class="flex flex-1 justify-start items-center flex-nowrap">
    <ChatBubbleLeftIcon
      overrideClasses={`mr-3 flex-shrink-0 h-5 w-5`}
    />
    {#if isEditing}
      <input
        bind:value={titleInput}
        on:click={(e) => e.stopPropagation()}
        type="text"
        name="title"
        class="block bg-gray-100 text-black w-full h-5 ring mr-3 rounded-md py-1.5 shadow-sm outline-0 sm:text-sm sm:leading-6"
      />
    {:else}
      <span
        class="w-28 text-left overflow-hidden whitespace-nowrap truncate "
        {title}
      >
        {truncateString(title)}
      </span>
    {/if}
  </div>

  <!-- Actions -->
  {#if isEditing}
    <div class="flex gap-2">
      <button on:click={() => handleSaveTitleEditClick()}>
        <CheckIcon overrideClasses={`h-3.5 w-3.5`} />
      </button>
      <button on:click={() => handleCancelTitleEditClick()}>
        <XMarkIcon overrideClasses={`h-3.5 w-3.5`} />
      </button>
    </div>
  {:else if isHovering}
    <div class="flex gap-2">
      <button on:click={() => handleTitleEditClick()}>
        <PencilSquareIcon
          overrideClasses={`h-3.5 w-3.5`}
        />
      </button>
      <button on:click={() => handleDeleteChat(chatId)}>
        <TrashIcon overrideClasses={`h-3.5 w-3.5`} />
      </button>
    </div>
  {/if}
</button>
