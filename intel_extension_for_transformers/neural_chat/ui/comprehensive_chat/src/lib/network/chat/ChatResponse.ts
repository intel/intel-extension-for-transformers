import { env } from '$env/dynamic/public';
import { fetchMsg } from '$lib/network/image/Network';
import { MessageType, type Message, MessageRole, isImage } from '$lib/shared/constant/Interface';

const GETIMAGELIST_URL = env.GETIMAGELIST_URL;
const UPLOAD_IMAGE_URL = env.UPLOAD_IMAGE_URL;
const BASE_URL = env.BASE_URL;


async function chatMessage(data: Message[]) {
	let i = data.length - 2
	for (; i >= 0; i--) {
		if (!(data[i].role === MessageRole.User && isImage(data[i].type))) break;
	}

	type img = {imgSrc: string;imgId: string;}
	let imageListBetween: img[] = []
	for (let item of data.slice(i + 1, data.length - 1)) {
		if (item.type === MessageType.SingleImage) imageListBetween = [...imageListBetween, item.content as img]
		else imageListBetween = [...imageListBetween, ...(item.content as img[])]
	}
	
	if (imageListBetween.length === 0) {
		let result = {
			query: data[data.length - 1].content
		}
		return fetchMsg('/chatWithImage', result)
	} else {
		let result = {
			query: data[data.length - 1].content,
			ImageList: imageListBetween,
		};
		return fetchMsg('/image2Image', result)
	}
}


async function UploadImage(data: any) {
	const url = `${UPLOAD_IMAGE_URL}`
}



export default { chatMessage };
