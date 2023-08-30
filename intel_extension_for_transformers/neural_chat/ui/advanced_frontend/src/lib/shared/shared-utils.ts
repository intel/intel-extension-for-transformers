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
    selected: {
      Model: string,
      "knowledge base": string,
    },
    MODEL_OPTION: ModelOptionType,
    article: string
  ) {
  let type: any = {};
  let articles:string[] = [];

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
    const knowledge_map: {[key:string]: string} = {"Wikipedia": "WIKI", "INC Document": "INC", "Customized": "Customized"}
    type = {
      model: "knowledge",
      knowledge: knowledge_map[selected["knowledge base"]],
    };
    if (selected["knowledge base"] === "Customized") {
      articles = article.split("\n");
    }
  }

  return [type, articles];
}

let chat: Record<string, Chat>;
let chatlist: ChatListItem[];

export const createNewChatListItem = (chatId: string): ChatListItem => {
  return {
    chatId,
    title: 'New chat'
  };
};

export const createNewChat = (
  chatId: string,
  messages: Message[],
  mode: string,
  optionType: string,
  selected: SelectedType,
  MODEL_OPTION: ModelOptionType,
  article: string,
  api_key: string
): Chat => {
  return {
    chatId,
    messages,
    mode,
    optionType,
    selected,
    MODEL_OPTION,
    article,
    api_key
  };
};

/**
 * Insert new chat (For the root route)
 */
export const insertNewChat = (
  msgs: Message[],
  mode: string,
  optionType: string,
  selected: SelectedType,
  MODEL_OPTION: ModelOptionType,
  article: string,
  api_key: string
) => {
  const newChatId = nanoid(8);

  chatList$.update((chatList) => {
    chatList.unshift(createNewChatListItem(newChatId));
    return chatList;
  });
  chats$.update((chats) => {
    chats[newChatId] = createNewChat(newChatId, msgs, mode,
      optionType,
      selected,
      MODEL_OPTION,
      article,
      api_key
    );
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
  mode: string,
  optionType: string,
  selected: SelectedType,
  MODEL_OPTION: ModelOptionType,
  article: string,
  api_key: string
) => {
  chats$.update((chats) => {
    chats[id].messages = msgs;
    chats[id].mode = mode;
    chats[id].optionType = optionType;
    chats[id].selected = selected,
    chats[id].MODEL_OPTION = MODEL_OPTION,
    chats[id].article = article,
    chats[id].api_key = api_key

    return chats;
  });
  // Problem
  try {
    const unsubscribe = chats$.subscribe((value: Record<string, Chat>) => {
      chat = value;
    });
    localStorage.setItem(id, JSON.stringify(chat[id]));
    unsubscribe();
  } catch (e: any) {
    console.log('update chat error', e);
  }
};

export const upsertChat = (
  chatId: string,
  msgs: Message[],
  mode: string,
  optionType: string,
  selected: SelectedType,
  MODEL_OPTION: ModelOptionType,
  article: string,
  api_key: string
) => {
  if (!chatId) {
    chatId = insertNewChat(msgs, mode,
      optionType,
      selected,
      MODEL_OPTION,
      article,
      api_key
    );
  } else {
    updateChat(chatId, msgs, mode,
      optionType,
      selected,
      MODEL_OPTION,
      article,
      api_key
    );
  }

  return chatId;
};

export function scrollToBottom(scrollToDiv: HTMLElement) {
  setTimeout(function () {
    scrollToDiv.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }, 100);
}

export const truncateString = (str = '', cutLength = 18) => {
  const truncated = str?.substring?.(0, cutLength);

  return truncated?.length < str?.length ? `${truncated}...` : truncated;
};