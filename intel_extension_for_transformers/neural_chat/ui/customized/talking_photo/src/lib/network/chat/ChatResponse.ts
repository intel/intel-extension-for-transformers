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

import { env } from '$env/dynamic/public';
import { fetchMsg, fetchTextMsg, tmpVideo } from '$lib/network/image/Network';
import { fetchTextStream, fetchVideoText } from '$lib/network/chat/Network';

const GETIMAGELIST_URL = env.GETIMAGELIST_URL;
const UPLOAD_IMAGE_URL = env.UPLOAD_IMAGE_URL;
const BASE_URL = env.BASE_URL;


async function chatMessage(query: string, voice_id: string, knowledge_base_id: string,
	ImageList: { imgSrc: string; imgId: string; }[], isVideo: boolean, currentMode: string, videoMode: string, photoMode: string) {
	if (currentMode === 'Search') {
		const result = {
			query,
			knowledge_base_id,
			domain: "test",
		}

		return fetchMsg('/chatWithImage', result)
	}
	if (currentMode === 'Video' && ImageList.length !== 0) {
		if (videoMode === 'output') {
			const result = await fetchVideoText(query, knowledge_base_id);
			// return value;
			query = result;

		}
		const blob = await fetch(ImageList[ImageList.length - 1].imgSrc).then(r => r.blob());
		const url = await tmpVideo(query, blob, voice_id)
		return {
			type: 'video',
			url
		}
	}
	if (currentMode === 'Photo') {
		const result = {
			query,
			ImageList,
		};
		return fetchMsg('/image2Image', result)
	}

}

export default { chatMessage };
