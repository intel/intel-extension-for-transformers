import { env } from "$env/dynamic/public";

const KNOWLEDGE_GAUDI2_URL = env.KNOWLEDGE_GAUDI2_URL;
const KNOWLEDGE_A100_URL = env.KNOWLEDGE_A100_URL;

export async function fetchKnowledgeBaseId(file: Blob, fileName: string) {
  const url = `${KNOWLEDGE_GAUDI2_URL}/create`;
  const formData = new FormData();
  formData.append("file", file, fileName);
  const init: RequestInit = {
    method: "POST",
    body: formData,
  };

  try {
    const response = await fetch(url, init);
    if (!response.ok) throw response.status;
    return await response.json();
  } catch (error) {
    console.error("network error: ", error);
    return undefined;
  }
}

export async function fetchKnowledgeBaseId2(file: Blob, fileName: string) {
  const url = `${KNOWLEDGE_A100_URL}/upload`;
  const formData = new FormData();
  formData.append("file", file, fileName);
  const init: RequestInit = {
    method: "POST",
    body: formData,
  };

  try {
    const response = await fetch(url, init);
    if (!response.ok) throw response.status;
    return await response.json();
  } catch (error) {
    console.error("network error: ", error);
    return undefined;
  }
}

export async function fetchKnowledgeBaseIdByPaste(pasteUrlList: any, urlType: string | undefined) {
  let url = ''
  // if (urlType === '1') {
    url = `${KNOWLEDGE_GAUDI2_URL}/upload_link`;
  // } else if (urlType === '2') {
    // url = `${KNOWLEDGE_A100_URL}/upload_link`;
  // }
  const data = {
    link_list: pasteUrlList,
  };
  const init: RequestInit = {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  };

  try {
    const response = await fetch(url, init);
    if (!response.ok) throw response.status;
    return await response.json();
  } catch (error) {
    console.error("network error: ", error);
    return undefined;
  }
}
