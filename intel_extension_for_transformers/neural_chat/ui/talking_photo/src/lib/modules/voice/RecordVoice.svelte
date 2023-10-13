<script lang="ts">
	import { onDestroy } from "svelte";
	import { createEventDispatcher } from "svelte";
	import audioOff from "$lib/assets/voice/svg/voiceOff.svg";
    import voiceWave from '$lib/assets/voice/svg/voiceOn.svg'

	const dispatch = createEventDispatcher()

	let chunks: any[] = [];
	let audioRecorder: MediaRecorder | undefined = undefined;
	let recordOK: boolean = false;
	let isRecording: boolean = false;
	let audioSrc: any = null;
	let interval: number;
    let voiceTimer = 0
	const MAX_AUDIO_TIME: number = 60;

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

					audioRecorder.addEventListener("dataavailable", (event) => {
						chunks.push(event.data)
					});

					audioRecorder.addEventListener("stop", () => {
						if (voiceTimer < 10) {
							dispatch('fail')
						} else {
							const blob = new Blob(chunks, { type: "audio/mp3; codecs=opus" });
							audioSrc = window.URL.createObjectURL(blob);
							dispatch('done', {src: audioSrc})
						}
						chunks = []
						voiceTimer = 0;
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
		}
	}

	function updateTimer() {
		if (voiceTimer++ >= MAX_AUDIO_TIME) {
			toggleRecording();
		}
	}

	onDestroy(() => {
		clearInterval(interval);
	});
</script>

<div class="flex flex-col justify-center items-center rounded-md">
	<!-- Voice button -->
	<label class="swap w-28">
	
		<!-- this hidden checkbox controls the state -->
		<input type="checkbox" on:change={toggleRecording} class="hidden"/>
		
		<!-- volume on icon -->
		<img class="h-28 swap-on"
        class:hidden={!isRecording}
        src={voiceWave} alt=""/>
		<!-- volume off icon -->
		<img class="h-28 swap-off " 
        class:hidden={isRecording}
        src={audioOff} alt="" />

	</label>
	<span class="text-[#6578aa] text-sm">{displayTimer(voiceTimer)}</span>
	<span class="text-sm ">{isRecording ? 'Recording' : 'Record Voice'}</span>
</div>
