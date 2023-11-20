export enum MessageRole {
	Assistant, User
}

export enum MessageType {
	Text, SingleAudio, AudioList, SingleImage, ImageList
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