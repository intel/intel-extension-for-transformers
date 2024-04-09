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

    let dragNode: HTMLImageElement
    let dropZone: HTMLElement

    function handleDragStart(e: DragEvent) {
        e.dataTransfer!.dropEffect = "copy";
        let el = e.target as HTMLImageElement;

        e.dataTransfer!.setData("src", el.src);
        e.dataTransfer!.setData("id", el.dataset.id!);
    }

	function handleTouchStart(e: TouchEvent) {
        let el = e.target as HTMLImageElement
        dragNode = el.cloneNode(true) as HTMLImageElement
        dropZone = document.querySelector('#drop-zone') as HTMLElement

        dropZone.appendChild(dragNode)

        dragNode.style.position = 'absolute'
        dragNode.style.height = `${el.clientHeight}px`
        dragNode.style.width = `${el.clientWidth}px`
        dragNode.style.display = 'none';

    }
    function handleTouchMove(e: TouchEvent) {
        e.preventDefault()
        dragNode.style.display = 'block';
        let relativeX = e.touches[0].clientX - dropZone.getBoundingClientRect().x - dragNode.clientWidth / 2
        let relativeY = e.touches[0].clientY - dropZone.getBoundingClientRect().y - dragNode.clientHeight / 2
        dragNode.style.left = `${relativeX}px`;
        dragNode.style.top = `${relativeY}px`;
    }
    function handleTouchEnd() {
        let dragPos = dragNode.getBoundingClientRect()
        let dropPos = dropZone.getBoundingClientRect()
        
        if (dragPos.x > dropPos.x && dragPos.y > dropPos.y) {
            droppedObj.set({src: dragNode.src, id: dragNode.dataset.id})
        }
        dropZone.removeChild(dragNode)
        dragNode.remove()
    }
</script>

<div
    draggable=true 
    on:dragstart={handleDragStart}
    on:touchstart={handleTouchStart}
    on:touchmove={handleTouchMove}
    on:touchend={handleTouchEnd}
    class="h-full w-full"
>
	<slot></slot>
</div>