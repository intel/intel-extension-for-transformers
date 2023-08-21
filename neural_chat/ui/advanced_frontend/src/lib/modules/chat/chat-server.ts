import { writable } from 'svelte/store';

const { set } = writable({
	models: [],
});

const POCKETBASE_API_URL = "https://stablediffusion.eglb.intel.com";

export const chatServer: ChatAdapter = {
    async modelList(): Promise<String[]> {
        try {
            const currentData = await chat_request({
                path: "/list_models",
                method: "POST",
            });
            const models = currentData.models;
            set({
                models,
            });

            return models;
        } catch (error) {
            return [];
        }
    },

    async txt2img() {
        const imgSrc = await img_request({
            path: "/stablediffusion",
            method: "POST",
        })

        return imgSrc;
    }

};

async function chat_request({
    path,
    method = "GET",
    body = null,
    headers = {},
    body_stringify = true,
}: {
    path: string;
    method?: string;
    body?: any;
    headers?: any;
    body_stringify?: boolean,
}) {
    const url = POCKETBASE_API_URL + path;

    const init: RequestInit = {
        method,
        headers,
        ...(body ? (body_stringify ? { body: JSON.stringify(body) } : body) : {}),
    };
    const request = fetch(url, init).then((r) => r.json());
    return request
}

async function img_request({
    path,
    method = "POST",
    body = null,
    headers = {},
    body_stringify = true,
}: {
    path: string;
    method?: string;
    body?: any;
    headers?: any;
    body_stringify?: boolean,
}) {
    const url = POCKETBASE_API_URL + path;

    const init: RequestInit = {
        method,
        headers,
        ...(body ? (body_stringify ? { body: JSON.stringify(body) } : body) : {}),
    };
    const request = fetch(url, init).then((r) => r.json());

    return request
}


interface ErrorResponse {
    message?: string;
    code?: number;
}

