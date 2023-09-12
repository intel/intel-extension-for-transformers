<script lang="ts">
	import { createEventDispatcher, onMount } from "svelte";

	export let src: string;
    export let autoPlay = false;

    let dispatcher = createEventDispatcher()
    let audioEl: HTMLAudioElement
    let play = false;

    onMount(() => {
        audioEl.addEventListener('ended', () => {
            audioEl.currentTime = 0
            play = false
            dispatcher('ended')
        })
        if (autoPlay) {
            play = true
            handlePlayClick()
        }
    })

    function handlePlayClick() {
        if (audioEl) {
            if (play === true) {
                audioEl.play()
            } else {
                audioEl.pause()
            }
        }
    }
</script>


<audio class="hidden" bind:this={audioEl} {src} />

<div class="flex">
    <label class="swap">
  
        <!-- this hidden checkbox controls the state -->
        <input type="checkbox" bind:checked={play} on:change={handlePlayClick} />
        
        <!-- volume on icon -->
        <svg class="swap-on fill-current w-5 h-5" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg"><path d="M512 1024A512 512 0 1 1 512 0a512 512 0 0 1 0 1024z m3.008-92.992a416 416 0 1 0 0-832 416 416 0 0 0 0 832zM320 320h128v384H320V320z m256 0h128v384H576V320z" fill="#bcdbff"></path></svg>
        <!-- volume off icon -->
        <svg class="swap-off fill-current w-5 h-5" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg"><path d="M512 1024A512 512 0 1 1 512 0a512 512 0 0 1 0 1024z m3.008-92.992a416 416 0 1 0 0-832 416 416 0 0 0 0 832zM383.232 287.616l384 224.896-384 223.104v-448z" fill="#bcdbff"></path></svg>
    </label>

    <div class="bg-contain bg-left bg-repeat-round w-20 ml-2" class:audio={play} class:default={!play}></div>
</div>

<style>
    .default {
        background-image: url(../../components/talkbot/imgs/audio1.png)
    }
    .audio {
        animation-name: flowingAnimation;
        animation-duration: 3s;
        animation-iteration-count: infinite;
        animation-timing-function: linear;
    }

    @keyframes flowingAnimation {
        0% {
            background-image: url(../../components/talkbot/imgs/audio1.png)
        }

        50% {
            background-image: url(../../components/talkbot/imgs/audio2.png)
        }

        100% {
            background-image: url(../../components/talkbot/imgs/audio1.png)
        }
    }
</style>
