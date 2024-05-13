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

export function scrollToBottom(scrollToDiv: HTMLElement) {
    if (scrollToDiv) {
        setTimeout(
            () =>
                scrollToDiv.scroll({
                    behavior: "auto",
                    top: scrollToDiv.scrollHeight,
                }),
            100
        );
    }
}

export function getCurrentTimeStamp() {
    return Math.floor(new Date().getTime() / 1000)
}

export function fromTimeStampToTime(timeStamp: number) {
    return new Date(timeStamp * 1000).toTimeString().slice(0, 8)
}


export function formatTime(seconds) {
    const hours = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const remainingSeconds = String(seconds % 60).padStart(2, '0');
    return `${hours}:${minutes}:${remainingSeconds}`;
}

