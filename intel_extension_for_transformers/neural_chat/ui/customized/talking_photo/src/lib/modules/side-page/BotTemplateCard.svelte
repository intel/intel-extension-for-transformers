<script lang="ts">
	import VoiceButton from "$lib/assets/voice/svelte/VoiceButton.svelte";
	import stopRecordingIcon from "$lib/assets/voice/svelte/StopRecording.svelte";
	import Knowledge from "$lib/assets/knowledge/svelte/Knowledge.svelte";
	import {
		afterUpdate,
		createEventDispatcher,
		onDestroy,
		onMount,
	} from "svelte";
	import EditIcon from "$lib/assets/voice/svelte/Edit.svelte";
	import RobotIcon from "$lib/assets/customize/svelte/RobotIcon.svelte";
	import VoiceWave from "$lib/assets/voice/svelte/VoiceWave.svelte";
	import StopRecording from "$lib/assets/voice/svelte/StopRecording.svelte";
	import CreateTemplate from "./CreateTemplate.svelte";
	import EditTemplate from "./EditTemplate.svelte";
	import TemplateBook from "$lib/assets/chat/svelte/TemplateBook.svelte";

	export let name: string;
	export let avatar: string;
	export let audio: string;
	export let knowledge: string;
	export let avatar_name = "";
	export let voice_name = "";
	export let knowledge_name = "";
	export let index: number = -1;
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
		if (music) music.pause();
	});
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div
	class="sm:w-[15rem] relative"
	on:click
>
	{#if notLibrary}
		<div class="absolute right-0 top-0 flex flex-row items-center gap-2">
			<EditTemplate {avatar_name} {voice_name} {knowledge_name} {index}>
				<svg
					class="hover:opacity-70"
					on:click|capture={() => {
						dispatch("edit");
					}}
					viewBox="0 0 1024 1024"
					version="1.1"
					xmlns="http://www.w3.org/2000/svg"
					width="15"
					height="15"
					><path
						d="M34.155089 230.940227 9.17948 230.940227 9.17948 256.203386 9.17948 854.158012C9.17948 923.769568 65.248004 980.289737 134.081773 980.289737L927.938515 980.289737 952.914125 980.289737 952.914125 955.026579 952.914125 471.100561C952.914125 457.148105 941.732164 445.837402 927.938515 445.837402 914.144868 445.837402 902.962906 457.148105 902.962906 471.100561L902.962906 955.026579 927.938515 929.76342 134.081773 929.76342C92.797081 929.76342 59.130699 895.825847 59.130699 854.158012L59.130699 256.203386 34.155089 281.466543 598.93821 281.466543C612.731859 281.466543 623.91382 270.155842 623.91382 256.203386 623.91382 242.250928 612.731859 230.940227 598.93821 230.940227L34.155089 230.940227Z"
						fill="#bbbbbb"
					/><path
						d="M437.016339 593.503789 431.876019 600.104892 431.668623 608.505214 427.984924 757.709741 427.077935 794.446421 461.312335 782.146455 605.005395 730.519447 611.980762 728.013291 616.479561 722.067243 1003.181673 210.964228 1018.529978 190.678421 998.306108 175.379305 869.49174 77.932781 849.985487 63.176536 834.913446 82.53177 437.016339 593.503789ZM839.575373 118.395018 968.389739 215.841542 963.514174 180.256619 576.81206 691.359633 588.286225 682.907428 444.593165 734.534436 477.920574 758.971151 481.604275 609.766622 476.256559 624.768047 874.153664 113.79603 839.575373 118.395018Z"
						fill="#bbbbbb"
					/><path
						d="M891.217762 310.505713 920.474916 269.553252 808.309143 187.564266 779.051989 228.516725 891.217762 310.505713Z"
						fill="#bbbbbb"
					/></svg
				>
			</EditTemplate>
			<svg
				class="hover:opacity-70"
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
		</div>
	{/if}

	<div
		class="flex sm:w-[15rem] flex-col gap-4 rounded-2xl border p-5"
		style="
		background: url(&quot;https://imgur.com/vfA05CS.png&quot;) center center / cover no-repeat;"
	>
		<div class="flex flex-row justify-between w-full">
			<img
				class="h-12 w-12 rounded-full shadow-lg"
				src={avatar}
				alt="Bonnie"
			/>
			<div
				class="flex flex-col justify-center items-center rounded-full bg-white sm:px-5 py-1 px-2 text-center text-sm font-medium text-white focus:ring-4 focus:ring-blue-300
            "
			>
				<button
					class="flex items-center justify-center"
					on:click|stopPropagation={() => {
						handleAudioPlayer();
					}}
				>
					{#if play}
						<StopRecording extraClass="h-7 w-7" />
					{:else}
						<VoiceWave extraClass="w-7 h-7" />
					{/if}
				</button>
				<!-- <p class="text-xs">Voice</p> -->
			</div>
		</div>

		<div class="">
			<div class="text-left text-lg font-bold">{name}</div>
			<div
				class="flex flex-row items-center gap-3 text-[0.93rem] text-[#4e5666] "
				class:opacity-0={knowledge_name === 'default'}
			>
				<TemplateBook /> <span class="w-full truncate whitespace-nowrap text-ellipsis ">{knowledge_name}</span> 
			</div>
		</div>
	</div>
</div>
