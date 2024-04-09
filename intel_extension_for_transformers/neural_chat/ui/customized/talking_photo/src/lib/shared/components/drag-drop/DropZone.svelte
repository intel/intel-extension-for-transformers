<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

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