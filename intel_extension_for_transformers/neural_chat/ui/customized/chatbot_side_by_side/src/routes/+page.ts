import { browser } from '$app/environment';
import { LOCAL_STORAGE_KEY } from '$lib/shared/constant/Interface';

export const load = async () => {
  if (browser) {
    const chat1 = localStorage.getItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY1);
    const chat2 = localStorage.getItem(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2);
    
    return {
      chatMsg1: JSON.parse(chat1 || '[]'),
      chatMsg2: JSON.parse(chat2 || '[]')
    }
  }
};
