import { env } from "$env/dynamic/public";

const TRAFFIC_URL = env.TRAFFIC_URL;

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

export async function trafficHint() {
	const url = TRAFFIC_URL;
	const init: RequestInit = {
		method: "GET",
	};

	try {
		let response = await fetch(url, init);
		if (!response.ok) throw response.status;
		const data = await response.text();
        const regex = /Waiting:\s*(\d+)/;
        if (data) {
            const match = data.match(regex);

            if (match) {
                const waitingValue = parseInt(match[1], 10);
                // console.log('Waiting:', waitingValue);

                return waitingValue;
            } else {
                console.error('could not find "Waiting" value');
                return null;
            }
        }

	} catch (error) {
		console.error("network error: ", error);
		return undefined;
	}
}


// async function retryFetch(url: string, maxRetries = 5) {
//     for (let retry = 0; retry < maxRetries; retry++) {
//         try {
//             const response = await fetch(url);
//             if (response.ok) {
//                 return response.text();
//             }
//         } catch (error) {
//             console.error('Fetch error:', error);
//         }
//     }
//     // throw new Error('already max retry');
//     console.log('already max retry');
// }

// export async function trafficHint() {
//     const url = TRAFFIC_URL;

//     try {
//         const data = await retryFetch(url);

//         const regex = /Waiting:\s*(\d+)/;
//         if (data) {
//             const match = data.match(regex);

//             if (match) {
//                 const waitingValue = parseInt(match[1], 10);
//                 console.log('Waiting:', waitingValue);

//                 return waitingValue;
//             } else {
//                 console.error('can not find "Waiting" value');
//                 return null;
//             }
//         }

//     } catch (error) {
//         if (error instanceof TypeError) {
//             console.error('network error:', error);
//         } else {
//             console.error('error:', error);
//         }
//         return null;
//     }
// }
