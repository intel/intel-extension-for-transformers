<script lang="ts">
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";

    export let content: { imgSrc: string; imgId: string; }
</script>

<div class="relative">
    <PictureEnlarge
        imgSrc={content.imgSrc}
        enlargeClass={"w-3 h-3"}
        extraClass={"right-0 top-0"}
    />
    <img class="w-[6vw] h-[6vw] max-sm:w-[20vw] max-sm:h-[20vw]" src={content.imgSrc} alt="" />
</div>