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