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