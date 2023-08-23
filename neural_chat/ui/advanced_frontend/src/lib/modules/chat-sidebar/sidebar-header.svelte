<script lang="ts">
	export let mode: string;

	const title_map: {[key:string]: string} = {
		basic: 'Used For?',
		advanced: 'Q&A',
		chatgpt: 'ChatGPT',
	}
</script>

<div
	class="font-title-intel text-xl text-white mb-4 pb-4 border-b flex gap-4 overflow-hidden"
>
	<span class="text-lg font-bold">{title_map[mode]}</span>
</div>
