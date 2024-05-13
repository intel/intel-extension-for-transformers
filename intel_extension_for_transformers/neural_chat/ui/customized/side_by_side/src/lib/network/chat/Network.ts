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

const CHAT_GAUDI2_URL = env.CHAT_GAUDI2_URL;
const CHAT_A100_URL = env.CHAT_A100_URL;

export async function fetchTextNoStream(
	query: string,
	knowledge_base_id: string,
	id
) {
	const url = CHAT_GAUDI2_URL;
	console.log("query knowledge_base_id", query, knowledge_base_id);
	let requestId = "request_id" + id;

	const postData = {
		prompt: query,
		request_id: requestId,
		kb_id: knowledge_base_id,
		stream: false,
	};

	const init: RequestInit = {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(postData),
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

export async function fetchTextNoStream2(
	query: string,
	knowledge_base_id: string,
	id
) {
	const url = CHAT_A100_URL;
	let requestId = "request_id" + id;

	const postData = {
		inputs: [
			{
				name: "prompt",
				datatype: "BYTES",
				shape: [1],
				data: ["How many people will attend CES?"],
			},
			{
				name: "kb_id",
				datatype: "BYTES",
				shape: [1],
				data: [knowledge_base_id],
			},
			{
				name: "request_id",
				datatype: "BYTES",
				shape: [1],
				data: [requestId],
			},
		],
	};

	const init: RequestInit = {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(postData),
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
