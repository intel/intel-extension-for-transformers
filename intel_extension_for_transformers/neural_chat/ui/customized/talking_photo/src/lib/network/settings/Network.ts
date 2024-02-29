import { env } from "$env/dynamic/public";

const BASE_URL = env.BASE_URL;




export async function onCountdownEnd() {	
	const url = `${BASE_URL}/deleteUser`
    const init: RequestInit = {
        method: "POST",
		
    };
	
	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status
		return await response.json();
	} catch (error) {
		console.error('network error: ', error);
		return undefined
	}
}


