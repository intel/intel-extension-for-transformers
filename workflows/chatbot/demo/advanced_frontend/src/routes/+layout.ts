// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { browser } from '$app/environment';

import {
  chats$,
  chatList$,
} from '$lib/shared/shared.store';
import { LOCAL_STORAGE_KEY } from '$lib/shared/shared.type';

export const load = async () => {
  /**
   * Sync localStorage to stores
   */
  if (browser) {
    const chatList = localStorage.getItem(LOCAL_STORAGE_KEY.CHAT_LIST);

    // Chat list
    if (chatList) {
      const parsedChatList = JSON.parse(chatList);
      chatList$.set(parsedChatList);

      // Chats
      if (parsedChatList.length > 0) {
        parsedChatList.forEach((listItem: any) => {
          const chatId = listItem.chatId;
          // chats$ messages should already be present in localStorage, else ¯\_(ツ)_/¯
          const chat = localStorage.getItem(chatId);

          if (chat) {
            chats$.update((chats) => {
              return {
                ...chats,
                [chatId]: JSON.parse(chat)
              };
            });
          }
        });
      }
    }
  }

  return {};
};
