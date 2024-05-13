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
import { LOCAL_STORAGE_KEY } from '$lib/shared/constant/Interface';

function getMsg(Chat_key: string) {
  const chat = localStorage.getItem(Chat_key);
  const items = [
    { id: 1, content: [], time: "0s" },
    { id: 2, content: [], time: "0s" },
  ];
  if (chat) {
    const chatMessagesMap = JSON.parse(chat);
    items.forEach((item) => {
      if (chatMessagesMap[item.id]) {
        item.content = chatMessagesMap[item.id];
      }
    });
  }
  return {
    chatMsg: JSON.parse(chat || '{}'),
    chatItems: items,
  }
}

export const load = async () => {
  if (browser) {

    const Msg1 = getMsg(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY)
    const Msg2 = getMsg(LOCAL_STORAGE_KEY.STORAGE_CHAT_KEY2)
    return {
      Msg1: Msg1,
      Msg2: Msg2 
    }
  }
};
