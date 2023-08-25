<script lang="ts">
	import Chat from "$lib/modules/chat/chat.svelte";

	export let data;
</script>


<main class="h-screen max-h-screen w-screen max-w-screen font-intel font-sm">
	<div class="flex h-full w-full">
		{#key data.chatId}
			<Chat chatId={data.chatId} />
		{/key}
	</div>
</main>
