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
  cardClass: `w-full border-[0.1rem] border-solid ${borderColor} p-[0.5rem] text-white xl:p-[1rem] xl:border-[0.2rem]`,
  titleClass: "text-[1.1rem] xl:text-xl",
  subtitleClass: "text-[0.8rem] xl:text-sm",
  borderBClass: `border-b-[0.1rem] ${borderColor} pb-2 text-[1.1rem] xl:text-xl leading-tight xl:border-b-[0.2rem] `,
  latencyContainerClass: `w-full border-[0.1rem]  border-solid ${borderColor}`,
  latencyValueClass: `text-[1.1rem] xl:text-xl ${bgColor} p-2`,
  latencySubtitleClass: "py-1 text-[1.1rem] xl:text-xl text-white ",
  reductionClass: `${bgColor} p-2 `,
  reductionValueClass: "text-[1.5rem] xl:text-2xl",
  reductionSubtitleClass: "text-[0.7rem] xl:text-xs leading-tight ",
  borderClass: `border-[0.1rem] ${borderColor} px-2 text-xl font-normal xl:border-[0.2rem] `,
  latencySubContentClass: "text-xs text-white pt-2",
});


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
