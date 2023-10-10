<script lang="ts">
    import { CollectionType, TalkingVoiceCustom, currentVoice } from '$lib/shared/stores/common/Store';
    import { TalkingVoiceLibrary } from '$lib/shared/constant/Data';
	import { getNotificationsContext } from 'svelte-notifications';
	import RecordVoice from '$lib/modules/voice/RecordVoice.svelte';
	import UploadVoice from '$lib/modules/voice/UploadVoice.svelte';
	import TalkingVoiceCard from '$lib/modules/voice/TalkingVoiceCard.svelte';


	const { addNotification } = getNotificationsContext();


    function handleVoiceUpload(e: CustomEvent<any>) {
		TalkingVoiceCustom.update(options => {
			return [{ name: e.detail.fileName, audio: e.detail.src }, ...options];
		});
	}

	function handleVoiceRecord(e: CustomEvent<any>) {
		TalkingVoiceCustom.update(options => {
			return [{ name: 'New Record', audio: e.detail.src }, ...options];
		});
		
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
		class="h-full w-full sm:mx-5 xl:mx-20 p-5 mt-4"
	>
        <div class="flex flex-wrap flex-row mb-1 sm:mb-0 justify-between w-full">
            <h2 class="text-md leading-tight md:pr-0 mb-6 font-medium text-[#051F61]">My Voices</h2>
        </div>
        {#if $TalkingVoiceCustom.length === 0}
            <div class="flex p-2 gap-7">
                <RecordVoice on:done={handleVoiceRecord} on:fail={handleRecordFail} />
                <UploadVoice on:upload={handleVoiceUpload} />
            </div>
        {:else}
            <div class="flex gap-7 text-[#0F172A] py-4">
                <RecordVoice on:done={handleVoiceRecord} on:fail={handleRecordFail} />
                <UploadVoice on:upload={handleVoiceUpload} />
            </div>
			<div class="flex overflow-auto gap-7">
				{#each $TalkingVoiceCustom as opt, i}
					<button
						class="m-2 rounded"
						class:ring={$currentVoice.collection === CollectionType.Custom && $currentVoice.id === i}
						on:click={() => { currentVoice.set({collection: CollectionType.Custom, id: i}) }}
					>
						<TalkingVoiceCard {...opt} on:delete={() => handleVoiceDelete(i)}/>
					</button>
				{/each}
			</div>
        {/if}
        
        <div class="flex flex-wrap flex-row mb-1 sm:mb-0 justify-between w-full mt-4">
            <h2 class="text-md leading-tight md:pr-0 mb-6 font-medium text-[#051F61]">Voice Library</h2>
        </div>
        <div class="flex gap-7 text-[#0F172A] overflow-auto py-4 px-2 w-full">
            {#each TalkingVoiceLibrary as opt, i}
                <button
					class="rounded"
                    class:ring={$currentVoice.collection === CollectionType.Library && $currentVoice.id === i}
                    on:click={() => { currentVoice.set({collection: CollectionType.Library, id: i}) }}
                >
                    <TalkingVoiceCard {...opt} on:delete={() => handleVoiceDelete(i)}/>
                </button>
            {/each}
        </div>
	</div>
</div>
