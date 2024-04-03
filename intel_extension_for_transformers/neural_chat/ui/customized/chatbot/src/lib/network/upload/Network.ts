import { env } from "$env/dynamic/public";

const BACKEND_BASE_URL = env.BACKEND_BASE_URL;

export async function fetchKnowledgeBaseId(file: Blob, fileName: string) {
  const url = `${BACKEND_BASE_URL}/create`;
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
  const url = `${BACKEND_BASE_URL}/upload_link`;
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
