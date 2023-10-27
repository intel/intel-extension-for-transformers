<script lang="ts">
	import Scrollbar from "$lib/shared/components/scrollbar/Scrollbar.svelte";
	import PictureEnlarge from "$lib/shared/components/images/PictureEnlarge.svelte";
	import { Avatar } from "flowbite-svelte";

    export let content: { imgSrc: string; imgId: string; }[]
</script>
<Scrollbar
    className="max-h-28 overflow-auto"
    classLayout="grid grid-cols-3 gap-4"
>
    {#each content as image}
        <div class="relative">
            <PictureEnlarge
                imgSrc={image.imgSrc}
                enlargeClass={"w-3 h-3"}
                extraClass={"right-0 top-0"}
            />
            <Avatar src={image.imgSrc} />
        </div>
    {/each}
</Scrollbar>