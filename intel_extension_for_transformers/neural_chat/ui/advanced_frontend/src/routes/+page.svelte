<script lang="ts">
	import Chat from "$lib/modules/chat/chat.svelte";
</script>

<svelte:head>
	<title>Neural Chat</title>
	<meta name="description" content="Neural Chat" />
</svelte:head>

<main class="h-screen max-h-screen font-intel font-sm">
	<div class="flex h-full">
		<Chat />
	</div>
</main>
