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
	import EditIcon from "$lib/assets/image-info/svelte/EditIcon.svelte";

	export let value: string;
	export let extraClass: string = "";
	export let inputExtraClass: string = "";
    export let showIcon = false
    let inputElement: HTMLInputElement;

    function focusInput() {
    inputElement.focus();
  }

</script>

<div class={`relative inline-flex px-0.5  ${extraClass}`}>
    {#if showIcon}
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <label class="absolute h-2 w-2 top-0 right-0 cursor-pointer" for="inlineInput"  on:click={focusInput}>
            <EditIcon />
        </label>
    {/if}
    <!-- <p>{value}</p> -->
	<input 
        type="text"
		class="{inputExtraClass} focus:border-blue-500 focus:ring-blue-500 border-none bg-transparent"
		bind:value
		on:change
        on:click|preventDefault|stopPropagation
        bind:this={inputElement}
	/>
</div>

