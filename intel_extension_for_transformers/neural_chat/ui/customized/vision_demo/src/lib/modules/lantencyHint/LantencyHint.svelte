<script>
	import {
		commonVariables,
		cardContents,
	} from "$lib/shared/constant/constants";
	import { latencyWritable } from "$lib/shared/stores/common/Store";
	import IntelFooterLogo from "$lib/assets/svg/intelFooterLogo.svg";

	const {
		cardClass,
		titleClass,
		subtitleClass,
		borderClass,
		latencySubContentClass,
		borderBClass,
		latencyContainerClass,
		latencyValueClass,
		latencySubtitleClass,
		reductionClass,
		reductionValueClass,
		reductionSubtitleClass,
	} = commonVariables;
</script>

<div
	class="relative mx-[2.4rem] pt-[2rem] flex h-full w-[15%] flex-col items-center gap-[0.5rem] text-center font-light xl:gap-[1rem] xl:pt-[2.3rem]"
>
	<div class={cardClass}>
		<p class={titleClass}>{cardContents.title1}</p>
		<p class={subtitleClass}>{cardContents.subtitle1}</p>
	</div>

	<div class={cardClass}>
		<p class={borderBClass}>{cardContents.title2}</p>
		<p class="mt-2">
			<span class={borderClass}>{cardContents.bit}</span> bit format
		</p>
	</div>

	<div class={latencyContainerClass}>
		<p class={latencyValueClass}>
			{#if $latencyWritable !== "0"}
				<span>{$latencyWritable}</span>

				{cardContents.latencySubtitle}
			{/if}
		</p>
		{#if cardContents.lantencySubContent !== ""}
			<p class={latencySubContentClass}>{cardContents.lantencySubContent}</p>
		{/if}
		<p class={latencySubtitleClass}>{cardContents.lantencyContent}</p>
	</div>

	{#if cardContents.reduction !== ""}
		<div class={reductionClass}>
			<p>
				<span class={reductionValueClass}>{cardContents.reduction}</span> Reduction
			</p>
			<p class={reductionSubtitleClass}>{cardContents.reductionSubtitle}</p>
		</div>
	{/if}
	<img class="absolute bottom-[1.5rem] xl:bottom-[2rem] right-0 w-[3.3rem] xl:w-[3.5rem]" src={IntelFooterLogo} alt="Your SVG" />

</div>

