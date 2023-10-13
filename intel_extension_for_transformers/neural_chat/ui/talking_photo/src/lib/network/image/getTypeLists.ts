import { fetchTypeList } from "./Network";

export async function getTypeList() {
    let res = await fetchTypeList()
    return res.type_list
}

export async function checkProcessingImage() {
    let res = await fetchTypeList()
    return [res.process_status.processing_image === 0, res.type_list, res.prompt_list]
}