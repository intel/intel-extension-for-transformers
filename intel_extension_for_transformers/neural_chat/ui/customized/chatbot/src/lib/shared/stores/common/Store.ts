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
