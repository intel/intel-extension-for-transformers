<script lang="ts">
    import { droppedObj } from "$lib/shared/stores/common/Store";
	import { createEventDispatcher } from "svelte";
	import { onDestroy } from "svelte";
    
    let dropZone: HTMLElement
    const dispatcher = createEventDispatcher()

    const unsubscribe = droppedObj.subscribe(value => {
        if (value.id) {
            dispatcher('drop', value)
        }
    })

    function handleDragDrop(e: DragEvent) {
        e.preventDefault();
        dispatcher('drop', { src: e.dataTransfer!.getData("src"), id: e.dataTransfer!.getData("id") });
    }

    onDestroy(unsubscribe)
</script>

<div
    id="drop-zone"
    bind:this={dropZone}
    on:dragover={e => e.preventDefault()} 
	on:drop={handleDragDrop} 
    class="h-full w-full relative"
>
	<slot></slot>
</div>