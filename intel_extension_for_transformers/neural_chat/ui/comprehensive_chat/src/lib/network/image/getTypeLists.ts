import { fetchTypeList } from "./Network";

export async function getTypeList() {
    let res = await fetchTypeList()
    // let originTypeList = res.type_list as {[index: string]: {[index: string]: string}[]};
    // let newTypeList: {[index: string]: {[index: string]: string}} = {}
    // for(let key in originTypeList) {
    //     newTypeList[key] = originTypeList[key].reduce((prev, cur) => {
    //         let [key, value] = Object.entries(cur)[0]
    //         prev[key] = value
    //         return prev
    //     }, {})
    // }
    return res.type_list
}

export async function checkProcessingImage() {
    let res = await fetchTypeList()
    return [res.process_status.processing_image === 0, res.type_list]
}