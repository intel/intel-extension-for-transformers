import { chatList$, chats$ } from './shared.store';
import { nanoid } from "nanoid";
import {
  LOCAL_STORAGE_KEY,
  type Chat,
  type ChatListItem,
  type ModelOptionType,
  type SelectedType,
  type Message,
} from "./shared.type";

export function defineType(
  optionType: string,
  selected: SelectedType,
  MODEL_OPTION: ModelOptionType,
) {
  let type: any = {};
  let modeltype: any = {};

  if (optionType == "Model") {
    type = {
      model: selected['Model'],
      temperature: MODEL_OPTION.options.find(
        (option) => option.label === "Temperature"
      )?.value,
      max_new_tokens: MODEL_OPTION.options.find(
        (option) => option.label === "Max Tokens"
      )?.value,
      topk: MODEL_OPTION.options.find(
        (option) => option.label === "Top K"
      )?.value,
    }
  } else if (optionType == "knowledge base") {
    const knowledge_map: { [key: string]: string } = { "Wikipedia": "WIKI", "INC Document": "INC", "ASK_GM": "ASK_GM", "Young_Pat": "Young_Pat", "Customized": "Customized" }
    type = {
      model: "knowledge",
      knowledge: knowledge_map[selected["knowledge base"]],
    };
    if (selected["advance option"]) {
      type.advanceOption = selected["advance option"]
    }
  }

  return type;
}

let chat: Record<string, Chat>;
let chatlist: ChatListItem[];

export const createNewChatListItem = (chatId: string, title: string): ChatListItem => {
  return {
    chatId,
    title: title
  };
};

export const createNewChat = (
  chatId: string,
  messages: Message[],
): Chat => {
  return {
    chatId,
    messages,
  };
};

/**
 * Insert new chat (For the root route)
 */
export const insertNewChat = (
  msgs: Message[],
  title: string
) => {
  const newChatId = nanoid(8);

  chatList$.update((chatList) => {
    chatList.unshift(createNewChatListItem(newChatId, title));
    return chatList;
  });
  chats$.update((chats) => {
    chats[newChatId] = createNewChat(newChatId, msgs);
    return chats;
  });

  try {
    const unsubscribe_chatlist = chatList$.subscribe((value: ChatListItem[]) => {
      chatlist = value;
    });
    const unsubscribe_chats = chats$.subscribe((value: Record<string, Chat>) => {
      chat = value;
    });
    localStorage.setItem(
      LOCAL_STORAGE_KEY.CHAT_LIST,
      JSON.stringify(chatlist)
    );
    localStorage.setItem(newChatId, JSON.stringify(chat[newChatId]));

    unsubscribe_chatlist();
    unsubscribe_chats();
  } catch (e: any) { }

  return newChatId;
};
/**
 * Update chat
 */

export const updateChat = (
  id: string,
  msgs: Message[],
  title: string
) => {
  chats$.update((chats) => {
    chats[id].messages = msgs;
    return chats;
  });

  chatList$.update((chatList) => {
    chatList = chatList.map((chat) => {
      if (chat.chatId === id) {
        chat.title = title;
      }
      return chat;
    });
    return chatList;
  });
  // Problem
  try {
    const unsubscribe_chatlist = chatList$.subscribe((value: ChatListItem[]) => {
      chatlist = value;
    });
    const unsubscribe_chats = chats$.subscribe((value: Record<string, Chat>) => {
      chat = value;
    });
    localStorage.setItem(
      LOCAL_STORAGE_KEY.CHAT_LIST,
      JSON.stringify(chatlist)
    );
    localStorage.setItem(id, JSON.stringify(chat[id]));
    unsubscribe_chatlist();
    unsubscribe_chats();
  } catch (e: any) {
    console.log('update chat error', e);
  }
};

export const upsertChat = (
  chatId: string,
  msgs: Message[],
  title: string
) => {
  if (!chatId) {
    chatId = insertNewChat(msgs, title);
  } else {
    updateChat(chatId, msgs, title);
  }

  return chatId;
};

export function scrollToBottom(scrollToDiv: HTMLElement) {
  if (scrollToDiv) {
    setTimeout(() => scrollToDiv.scroll({
      behavior: "auto",
      top: scrollToDiv.scrollHeight,
    }), 300)
  }
}

export const truncateString = (str = '', cutLength = 18) => {
  const truncated = str?.substring?.(0, cutLength);

  return truncated?.length < str?.length ? `${truncated}...` : truncated;
};