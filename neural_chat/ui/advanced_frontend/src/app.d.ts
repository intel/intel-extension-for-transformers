interface ChatAdapter {
	modelList(props: {
	}): Promise<Result<void>>;
	txt2img(props: {
	}): Promise<Result<void>>;
}

interface ChatMessage {
	role: string,
	content: string
}

interface ChatMessageType {
	model: string,
	knowledge: string,
	temperature: string,
	max_new_tokens: string,
	topk: string,
}