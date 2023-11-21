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

