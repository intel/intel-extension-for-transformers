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
