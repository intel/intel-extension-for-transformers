export interface ModelOptionType {
    names: string[];
    options: {
        label: string;
        value: number;
        minRange: number;
        maxRange: number;
        step: number;
        type: string;
    }[];
}

export interface SelectedType { Model: string, "knowledge base": string }

export enum LOCAL_STORAGE_KEY {
  OPEN_AI_API_KEY = 'openAiApiKey',
  CHAT_LIST = 'bChatList',
  GPT_MODEL = 'bGptModel'
}

export enum MESSAGE_ROLE {
  SYSTEM = 'system',
  ASSISTANT = 'assistant',
  USER = 'user',
  HUMAN = 'Human'
}

export enum BANNER_TYPE {
  ERROR = 'error',
  INFO = 'info',
  WARNING = 'warning',
  SUCCESS = 'success'
}

export enum ERROR {
  LOCAL_STORAGE_SET_ITEM = 'LOCAL_STORAGE_SET_ITEM',
  OPENAI_CHAT_COMPLETION = 'OPENAI_CHAT_COMPLETION',
  REGISTRATION = 'REGISTRATION',
  LOGIN = 'LOGIN',
  PASSWORD_RESET = 'PASSWORD_RESET',
  USER_DATA_FETCH = 'USER_DATA_FETCH',
  PASSWORD_CHANGE = 'PASSWORD_CHANGE',
  CHECKOUT_SESSION_CREATE = 'CHECKOUT_SESSION_CREATE',
  DATA_SYNC_SAVE = 'CHAT_SYNC_SAVE',
  DATA_SYNC_SAVE_LIMIT = 'CHAT_SYNC_SAVE_LIMIT',
  DATA_SYNC_IMPORT = 'CHAT_SYNC_IMPORT',
  DATA_SYNC_DELETE_SAVED_CHAT = 'CHAT_SYNC_DELETE_SAVED_CHAT'
}

export type Message = {
  role: MESSAGE_ROLE;
  content: string;
};

export type ChatListItem = {
  chatId: string;
  title: string;
};

export type Chat = {
  chatId: string;
  messages: Message[];
  mode: string,
  optionType: string,
  selected: { Model: string, "knowledge base": string },
  MODEL_OPTION: ModelOptionType,
  article: string,
  api_key: string
};


// In-memory only
export type Banner = {
  bannerId: string;
  bannerType: BANNER_TYPE;
  title: string;
  description: string;
};
