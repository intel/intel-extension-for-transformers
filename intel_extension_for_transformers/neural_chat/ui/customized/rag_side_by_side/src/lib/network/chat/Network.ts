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
