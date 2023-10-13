<script lang="ts">
	export let value: string;
	export let areaExtraClass: string = "";
</script>

<textarea
	rows="2"
	class="{areaExtraClass} w-full focus:border-blue-500 focus:ring-blue-500"
	bind:value
	on:change
/>
