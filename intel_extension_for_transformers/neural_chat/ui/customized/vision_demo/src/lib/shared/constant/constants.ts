// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// commonVariables.ts
interface CommonVariables {
  cardClass: string;
  titleClass: string;
  subtitleClass: string;
  borderBClass: string;
  latencyContainerClass: string;
  latencyValueClass: string;
  latencySubtitleClass: string;
  reductionClass: string;
  reductionValueClass: string;
  reductionSubtitleClass: string;
  borderClass: string;
  latencySubContentClass: string;
}

export const generateCommonVariables = (borderColor: string, bgColor:  string): CommonVariables => ({
  cardClass: `w-full border-[0.1rem] border-solid ${borderColor} p-[0.8rem] text-white`,
  titleClass: "text-[1.4rem] leading-5 py-[0.2rem]",
  subtitleClass: "text-[0.9rem] pt-[0.3rem]",
  borderBClass: `border-b-[0.1rem] ${borderColor} pb-[0.8rem] text-[1.4rem]  leading-tight `,
  latencyContainerClass: `relative w-full border-[0.1rem]  border-solid ${borderColor}`,
  latencyValueClass: `text-[1.1rem] ${bgColor}  h-[3rem]`,
  latencySubtitleClass: "pt-[0.5rem] pb-[1rem] text-[1.5rem] text-white ",
  reductionClass: `${bgColor} p-[1rem] `,
  reductionValueClass: "text-[2rem]",
  reductionSubtitleClass: "text-[0.8rem] leading-tight ",
  borderClass: `border-[0.1rem] ${borderColor} mr-[0.4rem] py-[0.1rem] px-[0.7rem] text-[1.3rem] font-normal`,
  latencySubContentClass: "text-[0.9rem] text-white block",
});

// export const generateCommonVariables = (borderColor: string, bgColor:  string): CommonVariables => ({
//   cardClass: `w-full border-[0.1rem] border-solid ${borderColor} p-[0.5rem] text-white xl:p-[1rem] xl:border-[0.2rem]`,
//   titleClass: "text-[1.1rem] xl:text-xl",
//   subtitleClass: "text-[0.8rem] xl:text-sm",
//   borderBClass: `border-b-[0.1rem] ${borderColor} pb-2 text-[1.1rem] xl:text-xl leading-tight xl:border-b-[0.2rem] `,
//   latencyContainerClass: `w-full border-[0.1rem]  border-solid ${borderColor}`,
//   latencyValueClass: `text-[1.1rem] xl:text-xl ${bgColor} p-2`,
//   latencySubtitleClass: "py-1 text-[1.1rem] xl:text-xl text-white ",
//   reductionClass: `${bgColor} p-2 `,
//   reductionValueClass: "text-[1.5rem] xl:text-2xl",
//   reductionSubtitleClass: "text-[0.7rem] xl:text-xs leading-tight ",
//   borderClass: `border-[0.1rem] ${borderColor} px-2 text-xl font-normal xl:border-[0.2rem] `,
//   latencySubContentClass: "text-xs text-white pt-2",
// });


// Intel® Xeon® 6
// export const commonVariables = generateCommonVariables("border-[#00c7fd]", "bg-[#00c7fd]");
// export const cardContents = {
// 	title1: "Intel® Xeon® 6",
// 	subtitle1: "with Performance Cores",
// 	title2: "Llama 2 70b",
// 	bit: 4,
// 	latency: "158",
// 	lantencyContent: "Latency",
// 	latencySubtitle: "ms",
// 	reduction: "6.4X",
// 	reductionSubtitle: "Next-Token Latency Versus 4ᵗʰ Gen Xeon® using 16 bit",
//   	lantencySubContent: "",
// };

// 5ᵗʰ Gen Intel® Xeon® Processor
// export const commonVariables = generateCommonVariables("border-[#6ddcff]", "bg-[#6ddcff]");
// export const cardContents = {
// 	title1: "5ᵗʰ Gen Intel® Xeon® Processor",
// 	subtitle1: "",
// 	title2: "Llama 2 70b",
// 	bit: 4,
// 	latency: "159",
// 	lantencyContent: "Latency",
// 	latencySubtitle: "ms",
// 	reduction: "3.6X",
// 	reductionSubtitle: "Next-Token Latency Versus 4ᵗʰ Gen Xeon® using 16 bit",
// 	lantencySubContent: "",
// };

// 4ᵗʰ Gen Intel® Xeon® Processor
// export const commonVariables =  generateCommonVariables("border-[#8bae46]", "bg-[#8bae46]");
// export const cardContents = {
// 	title1: "4ᵗʰ Gen Intel® Xeon® Processor",
// 	subtitle1: "",
// 	title2: "Llama 2 70b",
// 	bit: 4,
// 	latency: "192",
// 	lantencyContent: "Latency",
// 	latencySubtitle: "ms",
// 	reduction: "3X",
// 	reductionSubtitle: "Next-Token Latency Versus 16 bit format",
// 	lantencySubContent: "",
// };

// 4ᵗʰ Gen Intel® Xeon® Processor
export const commonVariables =  generateCommonVariables("border-[#b1d272]", "bg-[#b1d272]");

export const cardContents = {
	title1: "4ᵗʰ Gen Intel® Xeon® Processor",
	subtitle1: "",
	title2: "Llama 2 70b",
	bit: 16,
	latency: "566",
	lantencySubContent: "Next-Token",
	lantencyContent: "Latency",
	latencySubtitle: "ms",
	reduction: "",
	reductionSubtitle: "",
};
