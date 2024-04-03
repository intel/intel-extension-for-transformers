import { env } from "$env/dynamic/public";
import { SSE } from "sse.js";

const GNR_BACKEND_BASE_URL = env.GNR_BACKEND_BASE_URL;
const SPR_BACKEND_BASE_URL = env.SPR_BACKEND_BASE_URL;

export async function fetchTextStream(
	query: string,
	knowledge_base_id: string,
	group: string
) {
	console.log("1", query);

	let payload = {};
	let url = "";

	if (group == "1") {
		url = `${SPR_BACKEND_BASE_URL}/code_chat`;
		payload = {
			prompt: query,
			stream: true
		};
	} else if (group == "2") {
		url = `${GNR_BACKEND_BASE_URL}/code_chat`;
		payload = {
			prompt: query,
			stream: true
		};
		
	}

	return new SSE(url, {
		headers: { "Content-Type": "application/json" },
		payload: JSON.stringify(payload),
	});
}
