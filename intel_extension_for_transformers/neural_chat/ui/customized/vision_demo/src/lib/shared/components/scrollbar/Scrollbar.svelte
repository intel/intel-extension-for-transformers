<script lang="ts">
	import { Svroller } from "svrollbar";
    export let className: string = "";
    export let classLayout: string = "";
    export let alwaysVisible = true;
</script>

<div class={className}>
    <Svroller height="100%" width="100%" {alwaysVisible}>
        <div class={classLayout}>
            <slot></slot>
        </div>
    </Svroller>	
</div>

<style>
    :global(.svlr-contents) {
        height: 100%;
    }

    .row::-webkit-scrollbar {
		display: none;
	}

	.row {
		scrollbar-width: none;
	}

	.row {
		-ms-overflow-style: none;
	}
</style>