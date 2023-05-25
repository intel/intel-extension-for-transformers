import { chatServer } from "$lib/chat/chatServer";
import { SSE } from "sse.js";
import { env } from '$env/dynamic/public';

const LLMA_URL = env.LLMA_URL;
const GPT_J_6B_URL = env.GPT_J_6B_URL;
const KNOWLEDGE_URL = env.KNOWLEDGE_URL;

function chatMessage(chatMessages: ChatMessage[], type: ChatMessageType): SSE {
	// chatMessage
	const initWord =
		"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human questions.\n";
	let result = chatMessages.reduce((prev, cur) => {
		return prev + `${cur.role}: ${cur.content}\n`
	}, initWord)
	result += "Assistant:";

	const knowledgeContent = chatMessages[chatMessages.length - 1].content;
	
	const linkDict: {[key:string]: string} = {
		"llma": LLMA_URL,
		"gpt-j-6b": GPT_J_6B_URL,
		"knowledge": KNOWLEDGE_URL
	}

	const msgDataDict: {[key:string]: any} = {
		"llma": {
			model: "llama-7b-hf-conv",
			prompt: result,
			temperature: type.temperature,
			max_new_tokens: type.max_new_tokens,
			topk: type.topk, 
			stop: "<|endoftext|>",
		},
		"gpt-j-6b": {
			model: "gpt-j-6b",
			prompt: result,
			temperature: type.temperature,
			max_new_tokens: type.max_new_tokens,
			topk: type.topk, 
			stop: "<|endoftext|>",
		},
		"knowledge": {
			"query": knowledgeContent, 
			"domain": type.knowledge, 
			"debug": false,
		}
	}
	// request 
	const eventSource = new SSE(linkDict[type.model], {
			headers: { "Content-Type": "application/json" },
			payload: JSON.stringify( msgDataDict[type.model] ),
		}
	);

	return eventSource;
}

export default { modelList: chatServer.modelList, chatMessage };
