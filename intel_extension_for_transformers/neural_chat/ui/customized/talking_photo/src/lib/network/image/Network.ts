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

const BASE_URL = env.BASE_URL;

export async function uploadImages(image_list) {
	const url = `${BASE_URL}/uploadImages`;
	const init: RequestInit = {
		method: "POST",
		
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ image_list }),
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

export async function fetchUploadProgress(images) {
	const url = `${BASE_URL}/progress`;
	const init: RequestInit = {
		method: "GET",
		
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

export async function fetchImageList() {
	const url = `${env.BASE_URL}/getAllImages`
	const init: RequestInit = {
		method: "POST",
		
	}
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();
	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export async function tmpVideo(query: string | Blob, imageBlob: Blob, voice_id: string | Blob) {
	const url = `${env.VIDEO_URL}`;
	const formData = new FormData()
	formData.append('image', imageBlob, 'remote-image.jpg');
	formData.append('text', query);
	formData.append('mode', "fast");
	formData.append('voice', voice_id);

	const init: RequestInit = {
		method: "POST",
		body: formData,
	};

	try {
		const response = await fetch(url, init);
		if (!response.ok) throw response.status
		const videoData = await response.blob();

		const videoUrl = URL.createObjectURL(videoData);
		return videoUrl;
	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}

export async function fetchMsg(suffix, payload) {
	const url = `${env.BASE_URL}` + suffix;
	return sendPostRequest(url, payload);
}

// chat/knowldge 
export async function fetchTextMsg(suffix, payload) {
	const url = `${env.KNOWLEDGE_BASE_URL}` + suffix;
	return sendPostRequest(url, payload);
}

export async function fetchTypeList() {
	const url = `${env.BASE_URL}/getTypeList`;
	return sendPostRequest(url);
}

export async function fetchImageDetail(image_id: string) {
	const url = `${BASE_URL}/getImageDetail`;
	return sendPostRequest(url, { image_id });
}

export async function fetchImagesByType(type, subtype) {
	const url = `${BASE_URL}/getImageByType`;
	return sendPostRequest(url, { type, subtype });
}

export async function updateLabel(label, from, to) {
	const url = `${BASE_URL}/updateLabel`;
	return sendPostRequest(url, { label_list: [{ label, from, to }] });
}

export async function updateImageInfo(image_id, payload, urlSuffix) {
	const url = `${BASE_URL}` + urlSuffix;
	let extractedObject;

	if (payload) {
		extractedObject = {
			image_list: [
				{
					image_id,
					...payload,
				},
			],
		};
	} else {
		extractedObject = {
			image_id,
		};
	}

	const init: RequestInit = {
		method: "POST",
		
		body: JSON.stringify(extractedObject),
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

async function sendPostRequest(url: string, payload: Object = {}) {
	try {
		const response = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(payload),
		});

		if (!response.ok) throw response.status;

		return await response.json();
	} catch (error) {
		console.error("network error: ", error);
		return undefined;
	}
}
