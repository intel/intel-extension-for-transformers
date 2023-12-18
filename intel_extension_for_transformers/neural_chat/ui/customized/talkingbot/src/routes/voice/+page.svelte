<script lang="ts">
    import { CollectionType, TalkingVoiceCustom } from '$lib/components/talkbot/store';
    import { TalkingVoiceLibrary } from '$lib/components/talkbot/constant';
	import { getNotificationsContext } from 'svelte-notifications';
	import RecordVoice from '$lib/components/talkbot/record-voice.svelte';
	import UploadVoice from '$lib/components/talkbot/upload-voice.svelte';
	import TalkingVoiceCard from '$lib/components/talkbot/talking-voice-card.svelte';
    import { currentVoice } from '$lib/components/talkbot/store';
    import { fetchAudioEmbedding } from '$lib/modules/chat/network'

	const { addNotification } = getNotificationsContext();

	const customNum = $TalkingVoiceCustom.length
    let loading = false

    async function handleVoiceUpload(e: CustomEvent<any>) {
        loading = true
        let spk_id = "";
        try {
            const blob = await fetch(e.detail.src).then(r => r.blob());
            const res = await fetchAudioEmbedding(blob);
            spk_id = res.spk_id ? res.spk_id : "default";

        } catch {
            spk_id = "default";
        }
		TalkingVoiceCustom.update(options => {
			return [{ name: e.detail.fileName, audio: e.detail.src, id: spk_id }, ...options];
		});
        loading = false
	}

	async function handleVoiceRecord(e: CustomEvent<any>) {
        loading = true
        let spk_id = "";
        try {
            const blob = await fetch(e.detail.src).then(r => r.blob());
            const res = await fetchAudioEmbedding(blob)
            spk_id = res.spk_id ? res.spk_id : "default";
        } catch {
            spk_id = "default";
        }
		TalkingVoiceCustom.update(options => {
			return [{ name: 'New Record', audio: e.detail.src, id: spk_id }, ...options];
		});
        loading = false
	}

    function handleVoiceDelete(i :number) {
		TalkingVoiceCustom.update(options => {
			options.splice(i, 1)
			return options;
		});
	}
    
    function handleRecordFail() {
		addNotification({
			text: 'At least 10s required!',
			position: 'bottom-center',
			type: 'warning',
			removeAfter: 3000,
		});
	}
</script>

<div class="flex h-full">
	<div
		class="h-full w-full sm:mx-5 xl:mx-20 p-5 shadow"
	>
        <div class="flex flex-wrap flex-row mb-1 sm:mb-0 justify-between w-full">
            <h2 class="text-2xl leading-tight md:pr-0 mb-6 font-medium text-[#051F61]">My Voices</h2>
        </div>
        {#if $TalkingVoiceCustom.length === 0}
            <div class="flex p-2 gap-7">
                <RecordVoice on:done={handleVoiceRecord} on:fail={handleRecordFail} />
                <UploadVoice on:upload={handleVoiceUpload} />
                <span class="loading loading-bars loading-md text-[#8c75ff]" class:hidden={!loading}></span>
                <!-- <div class="flex flex-col justify-center items-start gap-5">
                    <h3 class="text-2xl">Create your own voices!</h3>
                </div> -->
            </div>
        {:else}
            <div class="flex gap-7 text-[#0F172A] py-4">
                <RecordVoice on:done={handleVoiceRecord} on:fail={handleRecordFail} />
                <UploadVoice on:upload={handleVoiceUpload} />
                <span class="loading loading-bars loading-md text-[#8c75ff]" class:hidden={!loading}></span>
                <div class="flex overflow-auto gap-7">
                    {#each $TalkingVoiceCustom as opt, i (opt.name + i)}
                        <button
                            class="m-2"
                            class:ring={$currentVoice.collection === CollectionType.Custom && $currentVoice.id === i}
                            on:click={() => { currentVoice.set({collection: CollectionType.Custom, id: i}) }}
                        >
                            <TalkingVoiceCard
                                audio={opt.audio}
                                bind:name={opt.name}
                                needChangeName={i >= customNum}
                                notLibrary
                                on:delete={() => handleVoiceDelete(i)}
                            />
                        </button>
                    {/each}
                </div>
            </div>
        {/if}
        
        <div class="flex flex-wrap flex-row mb-1 sm:mb-0 justify-between w-full mt-12">
            <h2 class="text-2xl leading-tight md:pr-0 mb-6 font-medium text-[#051F61]">Voice Library</h2>
            <!-- <div class="text-end">
                <form class="flex w-full space-x-3">
                    <div class=" relative ">
                        <input
                            type="text"
                            id="form-subscribe-Filter"
                            class=" rounded-lg border-transparent flex-1 appearance-none border border-gray-300 w-full py-2 px-4 bg-white placeholder-gray-400 shadow-sm text-base focus:outline-none focus:ring-1 focus:ring-purple-600 focus:border-transparent"
                            placeholder="name"
                        />
                    </div>
                    <button
                        class="flex-shrink-0 px-4 py-2 text-base font-semibold text-white bg-gray-600 rounded-lg shadow-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-purple-200"
                        type="submit"
                    >
                        Filter
                    </button>
                </form>
            </div> -->
        </div>
        <div class="flex gap-7 text-[#0F172A] overflow-auto py-4 px-2">
            {#each TalkingVoiceLibrary as opt, i (opt.name + i)}
                <button
                    class:ring={$currentVoice.collection === CollectionType.Library && $currentVoice.id === i}
                    on:click={() => { currentVoice.set({collection: CollectionType.Library, id: i}) }}
                >
                    <TalkingVoiceCard {...opt} on:delete={() => handleVoiceDelete(i)}/>
                </button>
            {/each}
        </div>
	</div>
</div>
