import { chatServer } from "$lib/modules/chat/chat-server";
import { SSE } from "sse.js";
import { env } from '$env/dynamic/public';

const AUDIO_URL = env.AUDIO_URL


async function fetchAudio(file, voice, knowledgeId) {
	const url = `${AUDIO_URL}`
	const formData = new FormData()
	formData.append('file', file)
	formData.append('voice', voice)
	formData.append('knowledge_id', knowledgeId)

    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.blob()
	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

async function fetchAudioText(file) {
	const url = `${AUDIO_URL}/asr`
	const formData = new FormData()
	formData.append('file', file)
    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();
	} catch (error) {
		console.error('network error: ', error);
		return error;
	}
}

async function fetchAudioStream(text, voice, knowledge_id) {
	const url = `${AUDIO_URL}/llm_tts`
	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify({ text, voice, knowledge_id }),
	})
}

export async function fetchAudioEmbedding(audio) {
	const url = `${AUDIO_URL}/create_embedding`
	const formData = new FormData()
	formData.append('file', audio)
    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();

	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export async function fetchKnowledgeBaseId(file) {
	const url = `${AUDIO_URL}/create_kb`
	const formData = new FormData()
	formData.append('file', file)
    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();

	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export default { modelList: chatServer.modelList, fetchAudio, fetchAudioText, fetchAudioStream };
