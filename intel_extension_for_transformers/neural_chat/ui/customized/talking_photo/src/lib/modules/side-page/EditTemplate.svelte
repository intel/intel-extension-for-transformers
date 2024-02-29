<script lang="ts">
	import { getNotificationsContext } from "svelte-notifications";
	import {
		TalkingPhotoCustom,
		TalkingVoiceCustom,
		TalkingKnowledgeCustom,
		TemplateCustom,
	} from "$lib/shared/stores/common/Store";
	import {
		TalkingPhotoLibrary,
		TalkingVoiceLibrary,
		TalkingKnowledgeLibrary
	} from "$lib/shared/constant/Data";
	import TalkingPhotoCard from "$lib/shared/components/talkbot/talking-photo-card.svelte";
	import TalkingVoiceCard from "$lib/shared/components/talkbot/TalkingVoiceCard.svelte";
	import TalkingKnowledgeCard from "$lib/shared/components/talkbot/talking-knowledge-card.svelte";

	import UploadAvatar from "$lib/shared/components/talkbot/upload-avatar.svelte";
	import RecordVoice from "$lib/shared/components/talkbot/RecordVoice.svelte";
	import UploadVoice from "$lib/shared/components/talkbot/UploadVoice.svelte";
	import { fetchAudioEmbedding, fetchKnowledgeBaseId, fetchKnowledgeBaseIdByPaste } from "$lib/network/talkbot/Network";
	import UploadKnowledge from "$lib/shared/components/talkbot/upload-knowledge.svelte";
	import { onMount } from "svelte";
	import { GradientButton } from "flowbite-svelte";
	import PasteKnowledge from "$lib/shared/components/talkbot/PasteKnowledge.svelte";
	import Loading from "$lib/assets/chat/svelte/Loading.svelte";

	export let avatar_name: string;
	export let voice_name: string
	export let knowledge_name: string
	export let index: number

	const { addNotification } = getNotificationsContext();
	let loading = false;
	let qualityMode = true;
	let kbLoading = false;
	let KBounce = false;
	let imgBounce = false;
	let VoiceBounce = false;

	let insertModal: HTMLDialogElement;
	$: allPhotos = [...$TalkingPhotoCustom, ...TalkingPhotoLibrary];
	$: allVoices = [...$TalkingVoiceCustom, ...TalkingVoiceLibrary];
	$: allKnowledges = [...$TalkingKnowledgeCustom, ...TalkingKnowledgeLibrary];

    let selectAvatar = -1;
	let selectVoice = -1;
	let selectKnowledge = -1;

    onMount(() => {
        selectAvatar = allPhotos.findIndex(val => val.name === avatar_name);
		selectVoice = allVoices.findIndex(val => val.name === voice_name);
		selectKnowledge = allKnowledges.findIndex(val => val.name === knowledge_name);
    })

	function editTemplate() {
		TemplateCustom.update((options) => {
            options[index] = {
                name: allPhotos[selectAvatar].name,
                avatar: allPhotos[selectAvatar].avatar,
                audio: allVoices[selectVoice].audio,
                identify: allVoices[selectVoice].identify,
                knowledge: allKnowledges[selectKnowledge].id,
                avatar_name: allPhotos[selectAvatar].name,
                voice_name: allVoices[selectVoice].name,
                knowledge_name: allKnowledges[selectKnowledge].name
            }
			return options
		});
		selectAvatar = -1;
		selectVoice = -1;
		selectKnowledge = -1;
	}

	function handleAvatarUpload(e: CustomEvent<any>) {
		TalkingPhotoCustom.update((options) => {
			return [{ name: e.detail.fileName, avatar: e.detail.src }, ...options];
		});
		imgBounce = true;

		setTimeout(() => {
			imgBounce = false;
		}, 3000);
	}

	function handleAvatarDelete(i: number) {
		TalkingPhotoCustom.update((options) => {
			options.splice(i, 1);
			return options;
		});
	}

	async function handleVoiceUpload(e: CustomEvent<any>) {
		loading = true;
		let spk_id = "";
		try {
			const blob = await fetch(e.detail.src).then((r) => r.blob());
			const res = await fetchAudioEmbedding(blob, qualityMode);
			spk_id = res.voice_id ? res.voice_id : "default";
			
		} catch {
			spk_id = "default";
		}
		TalkingVoiceCustom.update((options) => {
			return [
				{ name: e.detail.fileName, audio: e.detail.src, identify: spk_id },
				...options,
			];
		});
		loading = false;
		addNotification({
			text: "Uploaded successfully",
			position: "bottom-center",
			type: "success",
			removeAfter: 3000,
		});
		VoiceBounce = true;

		setTimeout(() => {
			VoiceBounce = false;
		}, 3000);
	}

	async function handleVoiceRecord(e: CustomEvent<any>) {
		loading = true;
		let spk_id = "";
		try {
			const blob = await fetch(e.detail.src).then((r) => r.blob());
			const res = await fetchAudioEmbedding(blob, qualityMode);
			spk_id = res.voice_id ? res.voice_id : "default";
		} catch {
			spk_id = "default";
		}
		TalkingVoiceCustom.update((options) => {
			return [
				{ name: "New Record", audio: e.detail.src, identify: spk_id },
				...options,
			];
		});
		loading = false;
		VoiceBounce = true;

		setTimeout(() => {
			VoiceBounce = false;
		}, 3000);
	}

	function handleRecordFail() {
		addNotification({
			text: "At least 10s required!",
			position: "bottom-center",
			type: "warning",
			removeAfter: 3000,
		});
	}

	async function handleKnowledgePaste(e: CustomEvent<any>) {
		let knowledge_id = "";
		try {
			const pasteUrlList = e.detail.pasteUrlList;
			const res = await fetchKnowledgeBaseIdByPaste(pasteUrlList);

			knowledge_id = res.knowledge_base_id ? res.knowledge_base_id : "default";
		} catch {
			knowledge_id = "default";
		}

		KBounce = true;

		setTimeout(() => {
			KBounce = false;
		}, 3000);
		addNotification({
			text: "Uploaded successfully",
			position: "top-left",
			type: "success",
			removeAfter: 3000,
		});

		TalkingKnowledgeCustom.update((options) => {
			return [
				{ name: "Knowledge Base", src: e.detail.src, id: knowledge_id },
				...options,
			];
		});
	}

	async function handleKnowledgeUpload(e: CustomEvent<any>) {
		kbLoading = true;
		let knowledge_id = "";
		try {
			const blob = await fetch(e.detail.src).then((r) => r.blob());
			const fileName = e.detail.fileName;
			const res = await fetchKnowledgeBaseId(blob, fileName);
			knowledge_id = res.knowledge_base_id ? res.knowledge_base_id : "default";
		} catch {
			knowledge_id = "default";
		}

		kbLoading = false;
		KBounce = true;

		setTimeout(() => {
			KBounce = false;
		}, 3000);

		addNotification({
			text: "Uploaded successfully",
			position: "bottom-center",
			type: "success",
			removeAfter: 3000,
		});

		TalkingKnowledgeCustom.update((options) => {
			return [
				{ name: e.detail.fileName, src: e.detail.src, id: knowledge_id },
				...options,
			];
		});
	}
	
	function handleKnowledgeDelete(i: number) {
		TalkingKnowledgeCustom.update((options) => {
			options.splice(i, 1);
			return options;
		});
	}
</script>

<button
	on:click={() => insertModal.showModal()}
>
	<slot>
		<div class="flex w-full aspect-video  rounded-3xl cursor-pointer flex-col justify-center rounded-md border-2 p-2">
			<svg
				class="mx-auto h-10 w-10"
				viewBox="0 0 1024 1024"
				version="1.1"
				xmlns="http://www.w3.org/2000/svg"
				fill="currentColor"
				><path
					d="M1022.955204 556.24776c0 100.19191-81.516572 181.698249-181.718715 181.698249l-185.637977 0c-11.2973 0-20.466124-9.168824-20.466124-20.466124 0-11.307533 9.168824-20.466124 20.466124-20.466124l185.637977 0c77.628008 0 140.786467-63.148226 140.786467-140.766001 0-77.423347-62.841234-140.448776-140.203182-140.766001-0.419556 0.030699-0.828878 0.051165-1.248434 0.061398-5.935176 0.153496-11.665691-2.302439-15.666818-6.702656-4.001127-4.41045-5.884011-10.345626-5.157463-16.250102 1.330298-10.806113 1.944282-19.760043 1.944282-28.192086 0-60.763922-23.658839-117.874641-66.617234-160.833035-42.968627-42.958394-100.089579-66.617234-160.843268-66.617234-47.368844 0-92.742241 14.449084-131.208321 41.781592-37.616736 26.738991-65.952084 63.700811-81.925894 106.884332-2.425236 6.54916-8.012488 11.399631-14.827707 12.893658-6.815219 1.483794-13.927197-0.603751-18.859533-5.536087-19.289322-19.340487-44.943608-29.982872-72.245418-29.982872-56.322773 0-102.146425 45.813419-102.146425 102.125959 0 0.317225 0.040932 0.982374 0.092098 1.627057 0.061398 0.920976 0.122797 1.831718 0.153496 2.762927 0.337691 9.465582-5.863545 17.928325-15.001669 20.455891-32.356942 8.943696-61.541635 28.550243-82.181721 55.217602-21.305235 27.516704-32.571836 60.508096-32.571836 95.41307 0 86.244246 70.188572 156.422585 156.443052 156.422585l169.981393 0c11.2973 0 20.466124 9.15859 20.466124 20.466124 0 11.2973-9.168824 20.466124-20.466124 20.466124l-169.981393 0c-108.828614 0-197.3753-88.536452-197.3753-197.354833 0-44.053332 14.223956-85.712127 41.126676-120.473839 22.809495-29.450752 53.897537-52.086285 88.710414-64.816215 5.065366-74.322729 67.149353-133.2447 142.751215-133.2447 28.386514 0 55.504128 8.217149 78.651314 23.52581 19.657712-39.868009 48.842405-74.169233 85.497233-100.212376 45.434795-32.295544 99.004875-49.354058 154.918325-49.354058 71.692832 0 139.087778 27.915793 189.782368 78.600149 50.694589 50.694589 78.610382 118.089535 78.610382 189.782368 0 3.704368-0.102331 7.470135-0.296759 11.368932C952.633602 386.245901 1022.955204 463.188294 1022.955204 556.24776z"
				/><path
					d="M629.258611 589.106122c-3.990894 3.990894-9.230222 5.996574-14.46955 5.996574s-10.478655-2.00568-14.46955-5.996574l-67.087954-67.077721 0 358.689289c0 11.307533-9.15859 20.466124-20.466124 20.466124-11.307533 0-20.466124-9.15859-20.466124-20.466124l0-358.689289-67.087954 67.077721c-7.992021 7.992021-20.947078 7.992021-28.939099 0s-7.992021-20.957311 0-28.949332l102.023628-102.013395c7.992021-7.992021 20.947078-7.992021 28.939099 0l102.023628 102.013395C637.250632 568.148811 637.250632 581.114101 629.258611 589.106122z"
				/></svg
			>
			<p class="text-sm text-black-300">Edit Template</p>
		</div>
	</slot>
</button>

<dialog class="max-sm:max-h-[90vh] max-sm:w-[90vw] p-5" bind:this={insertModal}>
    <div class="mb-7 overflow-auto">
        <h3 class="my-2 text-[1.2rem] mb-4 leading-tight text-[#051F61]">Talking Avatar</h3>
        <div class="grid grid-cols-4 gap-5 text-[#0F172A] sm:grid-cols-9">
			<UploadAvatar on:upload={handleAvatarUpload} />
            {#each allPhotos as opt, i (opt.name + i)}
                <button
                    class={`${i === 0 && imgBounce ? "animate-bounce" : ""}`}
                    class:ring={selectAvatar === i}
                    on:click={() => {selectAvatar = i}}
                >
                    <TalkingPhotoCard {...opt} />
                </button>
            {/each}
        </div>
    </div>
    <div class="mb-7">
        <h3 class="my-2 mb-6 text-[1.2rem] leading-tight text-[#051F61]">Talking Voice</h3>
        <div class="grid grid-cols-4 gap-3 text-[#0F172A] sm:grid-cols-9">
			<RecordVoice on:done={handleVoiceRecord} on:fail={handleRecordFail} />
			<UploadVoice on:upload={handleVoiceUpload} />
			{#if loading}
				<button class="aspect-square sm:w-[5rem] sm:h-[5rem] items-center ">
					<Loading />
				</button>
			{/if}
            {#each allVoices as opt, i (opt.name + i)}
                <button
					class="sm:w-[5rem] sm:h-[5rem]  aspect-square w-full {`${
						i === 0 && VoiceBounce ? 'animate-bounce' : ''
					}`}"
                    class:ring={selectVoice === i}
                    on:click={() => {selectVoice = i}}
                >
                    <TalkingVoiceCard {...opt} />
                </button>
            {/each}
        </div>
    </div>
	<div class="mb-7">
        <h3 class="my-2 mb-6 text-[1.2rem] leading-tight text-[#051F61]">Knowledge Base</h3>
        <div class="grid grid-cols-4 gap-2 text-[#0F172A] sm:grid-cols-9">
            <UploadKnowledge on:upload={handleKnowledgeUpload} />
			<PasteKnowledge on:paste={handleKnowledgePaste} />
			{#if kbLoading}
				<button class="aspect-square sm:w-[5rem] sm:h-[5rem] items-center">
					<Loading />
				</button>
			{/if}
            {#each allKnowledges as opt, i (opt.name + i)}
                <button
                    class:ring={selectKnowledge === i}
					class={`${i === 0 && KBounce ? "animate-bounce" : ""} sm:w-[5rem] sm:h-[5rem] `}
                    on:click={() => {selectKnowledge = i}}
                >
                    <TalkingKnowledgeCard {...opt} />
                </button>
            {/each}
        </div>
    </div>
	<div class="mt-10 flex justify-end gap-10 px-10">
		<GradientButton color="blue" on:click={() => {insertModal.close()}}>Cancel</GradientButton>
		<GradientButton color="blue" on:click={() => {editTemplate();insertModal.close()}}>Confirm</GradientButton>
    </div>
</dialog>
