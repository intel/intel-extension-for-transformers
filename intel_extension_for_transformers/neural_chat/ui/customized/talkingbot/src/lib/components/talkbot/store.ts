import { writable } from 'svelte/store';

export const TalkingPhotoCustom = writable<{
    name: string;
    avatar: string;
}[]>([])

export const TalkingVoiceCustom = writable<{
    name: string;
    audio: string;
    id: string;
}[]>([])

export const TalkingKnowledgeCustom = writable<{
    name: string;
    src: string;
    id: string;
}[]>([])

export const TalkingTemplateCustom = writable<{
    name: string;
    avatar: string;
    audio: string;
    knowledge: string;
}[]>([])

export enum CollectionType {
    Custom, Library, TemplateCustom, TemplateLibrary
}

export let currentAvaTar = writable<{
    collection: CollectionType,
    id: number
}>({
    collection: CollectionType.TemplateLibrary,
    id: 0
});
export let currentVoice = writable<{
    collection: CollectionType,
    id: number
}>({
    collection: CollectionType.TemplateLibrary,
    id: 0
});
export let currentKnowledge = writable<{
    collection: CollectionType,
    id: number
}>({
    collection: CollectionType.TemplateLibrary,
    id: 0
});
