import { env } from "$env/dynamic/public";

const BASE_URL = env.BASE_URL;

export async function uploadImages(image_list) {
	const url = `${BASE_URL}/uploadImages`;
	const init: RequestInit = {
		method: "POST",
		mode: 'cors',
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
		mode: 'cors',
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
	// return [
	// 	{
	// 		image_id: 27,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04172357736633.jpg",
	// 	},

	// 	{
	// 		image_id: 28,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04183253788089.jpg",
	// 	},

	// 	{
	// 		image_id: 34,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230911T06131561872402.jpg",
	// 	},

	// 	{
	// 		image_id: 46,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230912T02424874392762.jpg",
	// 	},
	// 	{
	// 		image_id: 28,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04183253788089.jpg",
	// 	},

	// 	{
	// 		image_id: 34,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230911T06131561872402.jpg",
	// 	},

	// 	{
	// 		image_id: 46,

	// 		image_path:
	// 			"http://54.147.152.170/ai_photos/user192.55.54.51/20230912T02424874392762.jpg",
	// 	},
	// ];
	const url = `${env.BASE_URL}/getAllImages`
	const init: RequestInit = {
		method: "POST",
		mode: 'cors',

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

export async function fetchMsg(suffix, payload) {
	const url = `${env.BASE_URL}` + suffix;
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
	// return [

	// 	{
	
	// 		"image_id": 27,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04172357736633.jpg"
	
	// 	},
	
	// 	{
	
	// 		"image_id": 28,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04183253788089.jpg"
	
	// 	},
	
	// 	{
	
	// 		"image_id": 34,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T06131561872402.jpg"
	
	// 	},


	// 	{
	
	// 		"image_id": 27,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04172357736633.jpg"
	
	// 	},
	
	// 	{
	
	// 		"image_id": 28,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T04183253788089.jpg"
	
	// 	},
	
	// 	{
	
	// 		"image_id": 34,
	
	// 		"image_path": "http://54.147.152.170/ai_photos/user192.55.54.51/20230911T06131561872402.jpg"
	
	// 	}
	
	// ]
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
		mode: 'cors',
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
			mode: 'cors',
			body: JSON.stringify(payload),
		});

		if (!response.ok) throw response.status;

		return await response.json();
	} catch (error) {
		console.error("network error: ", error);
		return undefined;
	}
}
