import { env } from "$env/dynamic/public";
import { SSE } from "sse.js";


const BASE_URL = env.BASE_URL;
const AUDIO_URL = env.AUDIO_URL;
const TALKING_URL = env.TALKING_URL;

export async function fetchAudioText(file) {
	const url = `${TALKING_URL}/asr`
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

export async function fetchAudioStream(text, voice, knowledge_id) {
	const url = `${AUDIO_URL}/llm_tts`
	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify({ text, voice, knowledge_id }),
	})
}

export async function fetchUploadProgress(images) {
	const url = `${BASE_URL}/progress`;
	const init: RequestInit = {
		method: "GET",
		// mode: 'no-cors',
	};

	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status;
		return await response.json();
	} catch (error) {
		console.error("network error: ", error);
		return undefined;
	}
}

