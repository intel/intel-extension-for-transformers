import { env } from "$env/dynamic/public";
import { SSE } from "sse.js";

const BACKEND_BASE_URL = env.BACKEND_BASE_URL;


export async function fetchTextStream(
	query: string,
	knowledge_base_id: string,
) {
	let payload = {};
	let url = "";

	payload = {
		prompt: query,
		stream: true
	};
	url = `${BACKEND_BASE_URL}/code_chat`;

	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify(payload),
	});
}
