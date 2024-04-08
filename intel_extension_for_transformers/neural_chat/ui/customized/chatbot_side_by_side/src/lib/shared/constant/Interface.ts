export enum MessageRole {
	Assistant, User
}

export enum MessageType {
	Text, SingleAudio, AudioList, SingleImage, ImageList, singleVideo
}


type Map<T> = T extends MessageType.Text | MessageType.SingleAudio ? string :
				T extends MessageType.AudioList ? string[] :
				T extends MessageType.SingleImage ? { imgSrc: string; imgId: string; } : 
				{ imgSrc: string; imgId: string; }[];

export interface Message {
	role: MessageRole,
	type: MessageType,
	content: Map<Message['type']>,
	time: number,
	first_token_latency: string,
	msecond_per_token: string
}

export enum LOCAL_STORAGE_KEY {
	STORAGE_CHAT_KEY1 = 'chatMessages1',
	STORAGE_CHAT_KEY2 = 'chatMessages2',
	STORAGE_TIME_KEY = 'initTime',
}