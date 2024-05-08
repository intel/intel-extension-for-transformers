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

import { writable } from "svelte/store";

export let open = writable(true);

export let knowledgeAccess = writable(true);

export let showTemplate = writable(false);

export let showSidePage = writable(false);

export let droppedObj = writable({});

export let isLoading = writable(false);

export let newUploadNum = writable(0);

export let ifStoreMsg = writable(true);

export const resetControl = writable(false);

export const knowledge1 = writable<{
	id: string;
}>();

export const knowledgeName = writable("");

export const latencyWritable = writable('0');