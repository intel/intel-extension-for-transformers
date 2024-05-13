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
    return [res.process_status.processing_image === 0, res.type_list, res.prompt_list]
}