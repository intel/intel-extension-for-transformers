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

import { env } from "$env/dynamic/public";
import { SSE } from "sse.js";


const BASE_URL = env.BASE_URL;
const TALKING_URL = env.TALKING_URL;
const TEXT_URL = env.KNOWLEDGE_BASE_URL;


export async function fetchTextStream(query: string, knowledge_base_id: string) {
	let payload = {};
	let url = "";

	console.log('knowledge_base_id', knowledge_base_id);
	
	if (knowledge_base_id !== "default") {
		payload = {
			"query": query,
			"domain": "test",
			"max_new_tokens": 128,
			knowledge_base_id,
			"stream": true

		}
		url = `${TEXT_URL}/chat`;

	} else {
		payload = {
			"query": query,
			"domain": "test",
			"stream": true,
			"max_new_tokens": 256,
			"knowledge_base_id": "default"
		}
		url = `https://198.175.88.26:443/v1/textchat/chat`;
	}

	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify(payload),
	})
}

export async function fetchVideoText(query: string, knowledge_base_id: string) {
	let payload = {};
	let url = "";

	if (knowledge_base_id !== "default") {
		payload = {
			"query": query,
			"domain": "test",
			"knowledge_base_id": knowledge_base_id,
			"max_new_tokens": 256,
			"stream": false
		}
		url = `${TEXT_URL}/chat`;

	} else {
		payload = {
			"query": query,
			"domain": "test",
			"knowledge_base_id": "default",
			"max_new_tokens": 256,
			"stream": false
		}
		url = `https://198.175.88.26:443/v1/textchat/chat`;
	}


	const init: RequestInit = {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(payload),
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

export async function fetchAudioText(file) {
	const url = `${TALKING_URL}/asr`
	const formData = new FormData()
	formData.append('file', file)
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

export async function fetchAudioStream(text: string, voice: any, knowledge_id: any) {
	const url = `${TALKING_URL}/llm_tts`
	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify({ text, voice, knowledge_id }),
	})
}

export async function fetchUploadProgress(images) {
	const url = `${BASE_URL}/progress`;
	const init: RequestInit = {
		method: "GET",

	};

	try {
		const response = await fetch(url, init);
		if (!response.ok) throw response.status;
		return await response.json();
	} catch (error) {
		console.error("network error: ", error);
		return undefined;
	}
}

