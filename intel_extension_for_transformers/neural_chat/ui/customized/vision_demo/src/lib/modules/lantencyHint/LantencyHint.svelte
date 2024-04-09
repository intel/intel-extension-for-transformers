<script>
	import {
		commonVariables,
		cardContents,
	} from "$lib/shared/constant/constants";
	import { latencyWritable } from "$lib/shared/stores/common/Store";
	import IntelFooterLogo from "$lib/assets/png/intelFooterLogo.png";
	import intelFooterLogoXeon from "$lib/assets/png/intelFooterLogoXeon.png";

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
	class="font-intel-light relative mx-[2.8rem] flex h-full w-[16.7%] flex-col items-center gap-[0.5rem] pt-[2.8rem] text-center font-light"
>
	<div class={cardClass}>
		<p class={titleClass}>{cardContents.title1}</p>
		<p class={subtitleClass}>{cardContents.subtitle1}</p>
	</div>

	<div class={cardClass}>
		<p class={borderBClass}>{cardContents.title2}</p>
		<p class="mt-[0.5rem]">
			<span class={borderClass}>{cardContents.bit}</span>
			<span class="text-[1.3rem]"> bit format</span>
		</p>
	</div>

	<div class={latencyContainerClass}>
		<p class={latencyValueClass}>
			{#if $latencyWritable !== "0"}
				<span class="text-[2rem]">{$latencyWritable}</span>

				{cardContents.latencySubtitle}
			{/if}
		</p>

		<p class={latencySubtitleClass}>
			{#if cardContents.lantencySubContent !== ""}
				<span class={latencySubContentClass}
					>{cardContents.lantencySubContent}</span
				>
			{/if}
			{cardContents.lantencyContent}
		</p>
	</div>

	{#if cardContents.reduction !== ""}
		<div class={reductionClass}>
			<p>
				<span class={reductionValueClass}>{cardContents.reduction}</span> Reduction
			</p>
			<p class={reductionSubtitleClass}>{cardContents.reductionSubtitle}</p>
		</div>
	{/if}
	<!-- <img class="absolute bottom-[4.3rem] right-0 w-[4rem]" src={IntelFooterLogo} alt="Your SVG" /> -->
	<!-- 6th -->
	<!-- <img class="absolute bottom-[4.3rem] right-0 w-[4rem]" src={intelFooterLogoXeon} alt="Your SVG" /> -->
	<!-- footer -->
	<img
		class="absolute bottom-[2.5rem] right-0 w-[4rem]"
		src={intelFooterLogoXeon}
		alt="Your SVG"
	/>
</div>
