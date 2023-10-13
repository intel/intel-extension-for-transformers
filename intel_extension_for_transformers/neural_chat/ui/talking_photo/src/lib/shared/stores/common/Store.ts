import { writable } from 'svelte/store';

export interface ImgInfo {
  image_id: number;
  image_path: string;
  caption: string;
  checked: boolean;
  location: string;
  time: string;
  tag_list: [string, string][];
};


// export interface ImgList {
//   [index: string]: {
//     [index: string]: ImgItem[]
//   }
// }

export interface ImgListPiece {
  image_id: string;
  image_path: string;
}

export let open = writable(true);

export let imageList = writable<Array<ImgListPiece>>([]);

export let droppedObj = writable({});

export let hintUploadImg = writable(true);

export let isLoading = writable(false);

export let newUploadNum = writable(0)

export let countDown = writable(0);

export let ifStoreMsg = writable(true)

export const resetControl = writable(false);


export const TalkingVoiceCustom = writable<{
  name: string;
  audio: string;
}[]>([])


export enum CollectionType {
  Custom, Library, TemplateCustom, TemplateLibrary
}


export let currentVoice = writable<{
  collection: CollectionType,
  id: number
}>({
  collection: CollectionType.TemplateLibrary,
  id: 0
});
