<script lang="ts">
	import { createEventDispatcher, onMount } from "svelte";

	export let src: string;
	export let autoPlay = false;

	let dispatcher = createEventDispatcher();
	let audioEl: HTMLAudioElement;
	let play = false;

	onMount(() => {
		audioEl.addEventListener("ended", () => {
			audioEl.currentTime = 0;
			play = false;
			dispatcher("ended");
		});
		if (autoPlay) {
			play = true;
			handlePlayClick();
		}
	});

	function handlePlayClick() {
		if (audioEl) {
			if (play === true) {
				audioEl.play();
			} else {
				audioEl.pause();
			}
		}
	}
</script>

<audio class="hidden" bind:this={audioEl} {src} />

<div class="flex">
	<label class="swap">
		<!-- this hidden checkbox controls the state -->
		<input
			type="checkbox"
			bind:checked={play}
			on:change={handlePlayClick}
			class="hidden"
		/>

		<!-- volume on icon -->
		<svg
			class="swap-on h-5 w-5 fill-current"
			class:hidden={!play}
			viewBox="0 0 1024 1024"
			version="1.1"
			xmlns="http://www.w3.org/2000/svg"
			><path
				d="M512 1024A512 512 0 1 1 512 0a512 512 0 0 1 0 1024z m3.008-92.992a416 416 0 1 0 0-832 416 416 0 0 0 0 832zM320 320h128v384H320V320z m256 0h128v384H576V320z"
				fill="#bcdbff"
			/></svg
		>
		<!-- volume off icon -->
		<svg
			class="swap-off h-5 w-5 fill-current"
			class:hidden={play}
			viewBox="0 0 1024 1024"
			version="1.1"
			xmlns="http://www.w3.org/2000/svg"
			><path
				d="M512 1024A512 512 0 1 1 512 0a512 512 0 0 1 0 1024z m3.008-92.992a416 416 0 1 0 0-832 416 416 0 0 0 0 832zM383.232 287.616l384 224.896-384 223.104v-448z"
				fill="#bcdbff"
			/></svg
		>
	</label>

	<div
		class="ml-2 w-20 bg-contain bg-left bg-repeat-round"
		class:audio={play}
		class:default={!play}
	/>
</div>

<style>
	.default {
		background-image: url(../../assets/chat/png/audio1.png);
	}
	.audio {
		animation-name: flowingAnimation;
		animation-duration: 3s;
		animation-iteration-count: infinite;
		animation-timing-function: linear;
	}

	@keyframes flowingAnimation {
		0% {
			background-image: url(../../assets/chat/png/audio1.png);
		}

		50% {
			background-image: url(../../assets/chat/png/audio2.png);
		}

		100% {
			background-image: url(../../assets/chat/png/audio1.png);
		}
	}

	.swap {
		position: relative;
		display: inline-grid;
		-webkit-user-select: none;
		user-select: none;
		place-content: center;
		cursor: pointer;
	}
	.swap > * {
		grid-column-start: 1;
		grid-row-start: 1;
		transition-duration: 0.3s;
		transition-timing-function: cubic-bezier(0, 0, 0.2, 1);
		transition-property: transform, opacity;
	}
</style>
