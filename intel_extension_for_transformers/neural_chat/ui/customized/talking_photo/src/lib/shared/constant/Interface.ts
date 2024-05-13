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

export enum MessageRole {
	Assistant, User
}

export enum MessageType {
	Text, SingleAudio, AudioList, SingleImage, ImageList, singleVideo
}

export const isAudio = (T: MessageType) => (T === MessageType.SingleAudio || T === MessageType.AudioList)
export const isImage = (T: MessageType) => (T === MessageType.SingleImage || T === MessageType.ImageList)

type Map<T> = T extends MessageType.Text | MessageType.SingleAudio ? string :
				T extends MessageType.AudioList ? string[] :
				T extends MessageType.SingleImage ? { imgSrc: string; imgId: string; } : 
				{ imgSrc: string; imgId: string; }[];

export interface Message {
	role: MessageRole,
	type: MessageType,
	content: Map<Message['type']>,
	time: number,
}

export enum LOCAL_STORAGE_KEY {
	STORAGE_CHAT_KEY = 'chatMessages',
	STORAGE_TIME_KEY = 'initTime',
}