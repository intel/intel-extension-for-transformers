<script lang="ts">
	export let src: string;
</script>

<!-- svelte-ignore a11y-media-has-caption -->
<video controls class="w-[10rem]">
	<source {src} type="video/mp4" />
	Your browser does not support the video tag.
</video>
