import { env } from '$env/dynamic/public';

const VOICE_FAST_URL = env.VOICE_FAST_URL;
const VOICE_HIGH_QUALITY_URL = env.VOICE_HIGH_QUALITY_URL;
const KNOWLEDGE_BASE_URL = env.KNOWLEDGE_BASE_URL;

export async function fetchKnowledgeBaseId(file: Blob, fileName: string) {
	const url = `${KNOWLEDGE_BASE_URL}/create`
	const formData = new FormData()
	formData.append('file', file, fileName)
    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		const response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();

	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export async function fetchKnowledgeBaseIdByPaste(pasteUrlList: any) {	
	const url = `${KNOWLEDGE_BASE_URL}/upload_link`
	const data = {
		"link_list": pasteUrlList,
	}
	const init: RequestInit = {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(data),
	};

	try {
		const response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();
	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export async function fetchAudioEmbedding(audio: Blob, qualityMode: boolean) {
	const url = qualityMode ? `${VOICE_FAST_URL}/create_embed` : `${VOICE_HIGH_QUALITY_URL}/create_embed`
	const formData = new FormData()
	formData.append('file', audio, "tmp.mp3")
    const init: RequestInit = {
        method: "POST",
        body: formData,
    };
	
	try {
		const response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();

	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}