<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

<script lang="ts">
	import chatResponse from "$lib/modules/chat/network";
	import stopRecordingIcon from "$lib/assets/stop-recording.svg";
    import VoiceButton from './imgs/voice-button.svelte'
	import { onDestroy } from "svelte";
	import VoiceWave from "./voice-wave.svelte";
	import { createEventDispatcher } from "svelte";
	import { MESSAGE_ROLE } from "../shared/shared.type";
	// Audio config
	export let chatMessages: any;

	const dispatch = createEventDispatcher()

	let chunks: any[] = [];
	let audioRecorder: MediaRecorder | undefined = undefined;
	let recordOK: boolean = false;
	let isRecording: boolean = false;
	let audioSrc: any = null;
	let interval: number;
    let voiceTimer = 0
	const MAX_AUDIO_TIME: number = 600;

	function pad(value: number) {
		return value.toString().padStart(2, "0");
	}

	function displayTimer(timer: number) {
		const minutes = pad(Math.floor(timer / 60));
		const seconds = pad(timer % 60);
		const time = `${minutes}:${seconds}`;
		return time;
	}

	async function getAudioPermission() {
		if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
			await navigator.mediaDevices
				.getUserMedia({ audio: true })
				.then((stream) => {
					audioRecorder = new MediaRecorder(stream);

					audioRecorder.addEventListener("dataavailable", (event) =>
						chunks.push(event.data)
					);

					audioRecorder.addEventListener("stop", async () => {
						dispatch('start')
						const blob = new Blob(chunks, { type: "audio/mp3; codecs=opus" });
						const res = await chatResponse.fetchAudioText(blob);
						console.log('res', res);
						
						audioSrc = window.URL.createObjectURL(blob);
						
						chatMessages = [...chatMessages, {
							role: MESSAGE_ROLE.USER,
							content: audioSrc,
							text: res.asr_result
						}]
						chunks = [];
						dispatch('done')
					});
					console.log("recordOK");
					recordOK = true;
				})
				.catch((error) => {
					console.error(`The following getUserMedia error occurred: ${error}`);
					recordOK = false;
				});
		} else {
			console.log("getUserMedia() is not supported in your browser");
			recordOK = false;
		}
	}

	async function toggleRecording() {
		if (!recordOK) {
			await getAudioPermission();
		}
		isRecording = !isRecording;
		if (isRecording) {
			console.log("Recording");
			audioRecorder?.start();
			interval = setInterval(updateTimer, 1000);
		} else {
			console.log("Stopped recording");
			audioRecorder?.stop();
			clearInterval(interval);
			voiceTimer = 0;
		}
	}

	function updateTimer() {
		if (voiceTimer++ > MAX_AUDIO_TIME) {
			toggleRecording();
		}
	}

	onDestroy(() => {
		clearInterval(interval);
	});
</script>

<VoiceWave running={isRecording}/>
<!-- Voice button -->
<button
	type="submit"
	on:click={toggleRecording}
>
	{#if isRecording}
		<img src={stopRecordingIcon} class="h-5 w-5" alt="" />
	{:else}
        <VoiceButton/>
	{/if}
</button>
<span>{displayTimer(voiceTimer)}</span>