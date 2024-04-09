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

<script>
	import {
		commonVariables,
		cardContents,
	} from "$lib/shared/constant/constants";
	import { latencyWritable } from "$lib/shared/stores/common/Store";
	import IntelFooterLogo from "$lib/assets/png/intelFooterLogo.png";

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
<!-- class="relative mx-[2.4rem] pt-[2rem] flex h-full w-[15%] flex-col items-center gap-[0.5rem] text-center font-light xl:gap-[1rem] xl:pt-[2.3rem]" -->

<div
class="relative mx-[2.8rem]  pt-[2.8rem] flex h-full w-[16.7%] flex-col items-center gap-[0.5rem] text-center font-light"
>
	<div class={cardClass}>
		<p class={titleClass}>{cardContents.title1}</p>
		<p class={subtitleClass}>{cardContents.subtitle1}</p>
	</div>

	<div class={cardClass}>
		<p class={borderBClass}>{cardContents.title2}</p>
		<p class="mt-[0.5rem]">
			<span class={borderClass}>{cardContents.bit}</span> <span class="text-[1.3rem]"> bit format</span> 
		</p>
	</div>

	<div class={latencyContainerClass}>
		<p class={latencyValueClass}>
			{#if $latencyWritable !== "0"}
				<span class="text-[2rem]">{$latencyWritable}</span>

				{cardContents.latencySubtitle}
			{/if}
		</p>
		{#if cardContents.lantencySubContent !== ""}
			<span class={latencySubContentClass}>{cardContents.lantencySubContent}</span>
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
	<img class="absolute bottom-[4.3rem] right-0 w-[3rem]" src={IntelFooterLogo} alt="Your SVG" />
	<!-- footer -->
	<!-- <img class="absolute bottom-[2.5rem] right-0 w-[3rem]" src={IntelFooterLogo} alt="Your SVG" /> -->


</div>

