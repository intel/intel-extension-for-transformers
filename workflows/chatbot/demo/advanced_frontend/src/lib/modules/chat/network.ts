import { chatServer } from "$lib/modules/chat/chat-server";
import { SSE } from "sse.js";
import { env } from '$env/dynamic/public';

const LLMA_URL = env.LLMA_URL;
const GPT_J_6B_URL = env.GPT_J_6B_URL;
const KNOWLEDGE_URL = env.KNOWLEDGE_URL;

function regFunc(currentMsg) {
	let text = currentMsg.slice(2, -1);
	const regex = /.*Assistant:((?:(?!",).)*)",/;
	const match = text.match(regex);
	let content = match ? match[1].trim() : "";
	content = content
		.replace(/\\\\n/g, "")
		.replace(/\\n/g, "")
		.replace(/\n/g, "")
		.replace(/\\'/g, "'");

	return content;
}

function chatMessage(chatMessages: ChatMessage[], type: ChatMessageType, articles = []): SSE {
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
			"articles": articles,
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

function chatGPT(msgs: ChatMessage[], api_key: string): SSE {
	return new SSE("https://api.openai.com/v1/chat/completions", {
			headers: { "Content-Type": "application/json", "Authorization": "Bearer " + api_key},
			payload: JSON.stringify({"model": "gpt-3.5-turbo", "messages": msgs}),
	})
}


export default { modelList: chatServer.modelList, chatMessage, chatGPT, regFunc };
