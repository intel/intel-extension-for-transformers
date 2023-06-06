<script lang="ts">
	import { createEventDispatcher } from "svelte";
	export let lists: Array<any>;
	export let title: string;
	export let selected = "";

	const dispatch = createEventDispatcher();

	$: {
		dispatch("change", {
			name: selected,
			list: lists[selected],
		});
	}
</script>

<div class="h-full">
	<div class="flex flex-col">
		<label class="flex flex-col items-start label">
			<div class="py-2 text-white">
				<span>Options</span>
			</div>
			<select
				class="select w-full bg-transparent text-white border-white"
				bind:value={selected}
			>
				{#each Object.keys(lists) as name, idx}
					<option class="bg-transparent text-black" value={name}>
						{name}
					</option>
				{/each}
			</select>
		</label>
	</div>

	<div
		class="shadow border-2 border-transparent"
		class:invisible={selected == ""}
		class:absolute={selected == ""}
	>
		{#if selected != ""}
			<div class="flex flex-col justify-start scroll-out overflow-hidden">
				{#each lists[selected] as option}
					<label class="label flex-col items-start bg-transparent text-white text-sm my-2">
						{#if option.type == 'range'}
							<span>{option.label}: {option.value}</span>
							<input
								class="w-full input-range bg-blue-400"
								type="range"
								max={option.maxRange}
								min={option.minRange}
								step={option.step}
								bind:value={option.value}
							/>
						{:else if option.type == 'file'}
							<span>{option.label}: </span>
							<input class="w-full input-range bg-blue-400 mt-5 border-none" type="file" on:change={(event) => {
								if (event?.target?.files.length == 0) return;
								let reader = new FileReader();
								reader.onload = (e) => {
									option.articles = e.target?.result;
								}
								reader.readAsText(event?.target?.files[0]);
							}}/>
						{/if}
					</label>
				{/each}
			</div>
		{/if}
	</div>
</div>

<style lang="postcss">
	.arrow {
		position: relative;
		&:after {
			position: absolute;
			display: block;
			height: 0.5rem;
			width: 0.5rem;
			--tw-translate-y: -100%;
			--tw-rotate: 45deg;
			transform: translate(var(--tw-translate-x), var(--tw-translate-y))
				rotate(var(--tw-rotate)) skew(var(--tw-skew-x)) skewY(var(--tw-skew-y))
				scaleX(var(--tw-scale-x)) scaleY(var(--tw-scale-y));
			transition-property: all;
			transition-duration: 0.15s;
			transition-duration: 0.2s;
			transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
			top: 50%;
			right: 1.4rem;
			content: "";
			transform-origin: 75% 75%;
			box-shadow: 2px 2px;
			pointer-events: none;
		}
	}

	.click-arrow {
		&:after {
			--tw-translate-y: -50%;
			--tw-rotate: 225deg;
		}
	}

	.scroll-out {
		/* animation-duration: 1s; */
		/* animation-name: scrollout; */
		/* animation-iteration-count: infinite; */
	}
	/* 
	@keyframes scrollin {
		from {
			margin-left: 100%;
		}

		to {
			margin-left: 0%;
		}
	} */
	@keyframes scrollout {
		from {
			height: 0;
			flex-grow: 0;
		}

		to {
			height: 0;
			flex-grow: 1;
		}
	}
</style>
