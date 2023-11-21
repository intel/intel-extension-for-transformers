<script lang="ts">
    import VoiceButton from './imgs/voice-button.svelte'
    import voiceWave from './imgs/Voice-Wave - Dark.svg'
  	import stopRecordingIcon from "$lib/assets/stop-recording.svg";  
	import { afterUpdate, createEventDispatcher, onDestroy, onMount } from "svelte";
	import EditIcon from "./imgs/Edit.svelte";
	import RobotIcon from "./imgs/Robot.svelte";

	export let name: string;
	export let avatar: string;
	export let audio: string;
    export let knowledge: string;
	export let notLibrary = false;
	export let needChangeName = false;

	const dispatch = createEventDispatcher();

	onMount(() => {
		music = new Audio(audio);
        music.onended = () => {
            music.currentTime = 0;
            play = false;
            music.pause();
        };
	});

    afterUpdate(() => {
		if (inputEl) {
			inputEl.focus();
			inputEl.onblur = () => {
				needChangeName = false;
			};
		}
	});

	let inputEl: HTMLInputElement;
	let play = false;
	let music: HTMLAudioElement;
	function handleAudioPlayer() {
		play = !play;
		if (play) {
			music.play();
		} else {
			music.pause();
		}
	}
    function changeName() {
		needChangeName = true;
	}

    onDestroy(() => {
        if (music) music.pause()
    })
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div class="relative w-full rounded-2xl shadow-[0_2px_30px_0_rgba(0,0,0,0.2)] p-2" on:click>
    {#if notLibrary}
		<svg
			class="absolute right-0 top-0 hover:opacity-70"
			on:click={() => {
				dispatch("delete");
			}}
			viewBox="0 0 1024 1024"
			version="1.1"
			xmlns="http://www.w3.org/2000/svg"
			width="20"
			height="20"
		>
			<path
				d="M512 832c-176.448 0-320-143.552-320-320S335.552 192 512 192s320 143.552 320 320-143.552 320-320 320m0-704C300.256 128 128 300.256 128 512s172.256 384 384 384 384-172.256 384-384S723.744 128 512 128"
				fill="#bbbbbb"
			/><path
				d="M649.824 361.376a31.968 31.968 0 0 0-45.248 0L505.6 460.352l-98.976-98.976a31.968 31.968 0 1 0-45.248 45.248l98.976 98.976-98.976 98.976a32 32 0 0 0 45.248 45.248l98.976-98.976 98.976 98.976a31.904 31.904 0 0 0 45.248 0 31.968 31.968 0 0 0 0-45.248L550.848 505.6l98.976-98.976a31.968 31.968 0 0 0 0-45.248"
				fill="#bbbbbb"
			/>
		</svg>
	{/if}
    <div class="p-5 pb-0">
        <div class="flex justify-between relative">
            <img class="w-24 h-24 cursor-pointer hover:border" src={avatar} alt="">
            <div class="w-7/12 pb-0">
                <div class="flex items-center gap-2 pt-2 relative">
                    <div class="w-8 h-8 rounded-full border-none p-0 flex justify-center items-center bg-white">
                        <div
                            class="flex items-center justify-center"
                            on:click|stopPropagation={() => {
                                handleAudioPlayer();
                            }}
                        >
                            {#if play}
                                <img class="my-1 h-11 w-11" src={stopRecordingIcon} alt="" />
                            {:else}
                                <img class="w-8 h-8" src={voiceWave} alt="">
                            {/if}
                        </div>
                    </div>
                    <div class="text-left text-base truncate font-bold opacity-70">
                        Sample Voice
                    </div>
                </div>
                <div class="flex items-center gap-2 pt-4 relative">
                    <div class="w-8 h-8 rounded-full border-none p-0 flex justify-center items-center bg-white" on:click>
                        <RobotIcon />
                    </div>
                    <div class="text-left text-base truncate font-bold opacity-70">
                        Start Chat
                    </div>
                </div>
            </div>
        </div>
        <div
            class="text-sm rounded-lg py-2 font-bold text-yellow-600 text-left"
        >
            <span class="relative">
                <!-- {#if needChangeName}
                    <input
                        type="text"
                        bind:value={name}
                        bind:this={inputEl}
                        class="w-full text-base text-gray-600 focus-visible:outline-[#ccc]"
                    />
                {:else} -->
                    <span>{name}</span>
                    <!-- {#if notLibrary}
                        <span class="absolute -right-3 -top-1"><EditIcon on:changeName={changeName} /></span>
                    {/if}
                {/if} -->
            </span>
        </div>
    </div>
</div>
