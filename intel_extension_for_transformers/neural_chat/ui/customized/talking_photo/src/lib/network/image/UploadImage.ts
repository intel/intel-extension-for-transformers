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

import { uploadImages } from "$lib/network/image/Network";
import { imageList, type ImgListPiece } from "$lib/shared/stores/common/Store";

export async function handleImageUpload(e: CustomEvent<any>, resolve:Function = ()=>{}) {
    const files = e.detail.blobs
    let image_list: { imgSrc: string }[] = []

    for (let i = 0; i < files.length; ++i) {
        const reader = new FileReader();
        reader.onloadend = async () => {
            if (!reader.result) return;
            const src = reader.result.toString();            

            image_list = [...image_list, { imgSrc: src }]
            if (image_list.length === files.length) {
                await uploadImageList(image_list)
                resolve()
            }
        };
        reader.readAsDataURL(files[i]);
    }

}

async function uploadImageList(image_list: { imgSrc: string }[]) {
    const uploadRes = await uploadImages(image_list)
    
    const combinedArray: ImgListPiece[] = uploadRes.map((info) => ({        
        image_id: info.img_id,
        image_path: info.img_path
      }));
    
    imageList.update(imageListArray => [...imageListArray, ...combinedArray]);       
}

