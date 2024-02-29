<script lang="ts">
	import { getNotificationsContext } from "svelte-notifications";
	import {
		TalkingPhotoCustom,
		TalkingVoiceCustom,
		TalkingKnowledgeCustom,
		TemplateCustom,
		showTemplate,
		currentTemplate,
		CollectionType,
	} from "$lib/shared/stores/common/Store";
	import {
		TalkingPhotoLibrary,
		TalkingVoiceLibrary,
		TalkingKnowledgeLibrary,
	} from "$lib/shared/constant/Data";
	import TalkingPhotoCard from "$lib/shared/components/talkbot/talking-photo-card.svelte";
	import TalkingVoiceCard from "$lib/shared/components/talkbot/TalkingVoiceCard.svelte";
	import TalkingKnowledgeCard from "$lib/shared/components/talkbot/talking-knowledge-card.svelte";

	import UploadAvatar from "$lib/shared/components/talkbot/upload-avatar.svelte";
	import RecordVoice from "$lib/shared/components/talkbot/RecordVoice.svelte";
	import UploadVoice from "$lib/shared/components/talkbot/UploadVoice.svelte";
	import {
		fetchAudioEmbedding,
		fetchKnowledgeBaseId,
		fetchKnowledgeBaseIdByPaste,
	} from "$lib/network/talkbot/Network";
	import UploadKnowledge from "$lib/shared/components/talkbot/upload-knowledge.svelte";
	import Loading from "$lib/assets/chat/svelte/Loading.svelte";
	import { GradientButton } from "flowbite-svelte";
	import PasteKnowledge from "$lib/shared/components/talkbot/PasteKnowledge.svelte";

	let selectAvatar = -1;
	let selectVoice = -1;
	let selectKnowledge = -1;

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

	function addTemplate() {
		TemplateCustom.update((options) => {
			return [
				{
					name: allPhotos[selectAvatar].name,
					avatar: allPhotos[selectAvatar].avatar,
					audio: allVoices[selectVoice].audio,
					identify: allVoices[selectVoice].identify,
					knowledge: allKnowledges[selectKnowledge].id,
					avatar_name: allPhotos[selectAvatar].name,
					voice_name: allVoices[selectVoice].name,
					knowledge_name: allKnowledges[selectKnowledge].name,
				},
				...options,
			];
		});
		console.log($TemplateCustom);

		selectAvatar = -1;
		selectVoice = -1;
		selectKnowledge = -1;

		currentTemplate.set({
	            collection: CollectionType.Custom,
	            id: 0,
	        });
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

	function handleAvatarError() {
		addNotification({
			text: "Please upload valid photos!",
			position: "bottom-center",
			type: "success",
			removeAfter: 3000,
		});
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

	async function handleKnowledgePaste(e: CustomEvent<{pasteUrlList: string[]}>) {
		if (e.detail.pasteUrlList.some(el => {
			const regex = /^([\x09\x0A\x0D\x20-\x7E]|[\xC2-\xDF][\x80-\xBF]|\xE0[\xA0-\xBF][\x80-\xBF]|[\xE1-\xEC\xEE\xEF][\x80-\xBF]{2}|\xED[\x80-\x9F][\x80-\xBF]|\xF0[\x90-\xBF][\x80-\xBF]{2}|[\xF1-\xF3][\x80-\xBF]{3}|\xF4[\x80-\x8F][\x80-\xBF]{2})*$/;

			return regex.test(el);
		})) {
			addNotification({
				text: "Please upload valid links",
				position: "top-left",
				type: "success",
				removeAfter: 3000,
			});
			return 
		}
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
			position: "top-left",
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

<div class="overflow-auto h-full">
	<div class="mt-7 overflow-auto">
        <h3
            class="my-2  text-[1.2rem] mb-4 leading-tight text-[#051F61]"
        >
            Talking Avatar
        </h3>
        <div class="grid grid-cols-4 gap-5 text-[#0F172A] grid-cols-9">
            <UploadAvatar on:upload={handleAvatarUpload} on:error={handleAvatarError} />
            {#each allPhotos as opt, i (opt.name + i)}
                <button
                    class={`${i === 0 && imgBounce ? "animate-bounce" : ""} mb-2`}
                    class:ring={selectAvatar === i}
                    on:click={() => {
                        selectAvatar = i;
                    }}
                >
                    <TalkingPhotoCard {...opt} />
                </button>
            {/each}
        </div>
    </div> 
	<div class="mt-7 overflow-auto">
		<h3 class="my-2 mb-6 text-[1.2rem] leading-tight text-[#051F61]">
			Talking Voice
		</h3>
		<div class="grid grid-cols-4 gap-2 text-[#0F172A] sm:grid-cols-9 sm:p-2">
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
					on:click={() => {
						selectVoice = i;
					}}
				>
					<TalkingVoiceCard {...opt} />
				</button>
			{/each}
		</div>
	</div>
	
	<div class="mt-7 mb-16 overflow-auto">
		<h3 class="my-2 mb-6 text-[1.2rem] leading-tight text-[#051F61]">
			Knowledge Base
		</h3>
		<div class="grid grid-cols-4 gap-2 text-[#0F172A]  sm:grid-cols-9">
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
					on:click={() => {
						selectKnowledge = i;
					}}
				>
					<TalkingKnowledgeCard {...opt} />
				</button>
			{/each}
		</div>
	</div>
	<div class="mt-10 flex justify-end gap-10 px-10 fixed bottom-5 right-10">
		<GradientButton
			color="blue"
			on:click={() => {
				selectAvatar = selectVoice = selectKnowledge = -1;
                showTemplate.set(false);
        }}>Cancel</GradientButton
		>
		<GradientButton
			color="blue"
			on:click={() => {
				addTemplate();
                showTemplate.set(false);
			}}>Confirm</GradientButton
		>
	</div>
</div>
