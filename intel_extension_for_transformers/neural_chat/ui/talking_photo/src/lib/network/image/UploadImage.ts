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

