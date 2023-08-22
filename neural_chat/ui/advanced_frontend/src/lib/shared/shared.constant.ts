import improve from "$lib/assets/104b-add-to-favorites-outlined.svg";
import transport from "$lib/assets/transport.svg";
import financial from "$lib/assets/financial.svg";
import stock from "$lib/assets/stock.svg";
import education from "$lib/assets/education.svg";
import realEstate from "$lib/assets/real_estate.svg";

// chat  
export const DEFAULT_TIP = [
  {
    name: "What are the benefits of regular exercise?",
    icon: improve,
  },
  { name: "How does climate change impact the environment?", icon: improve },
  {
    name: "Please provide me with a hearty lunch menu.",
    icon: improve,
  },
];
export const TIPS_DICT = {
  Wikipedia: [
    { name: "What is the largest ocean in the world?", icon: improve },
    { name: "Who was the first person to walk on the moon?", icon: improve },
    { name: "What is the tallest mountain in the world?", icon: improve },
  ],
  "INC Document": [
    { name: "What is the quantization?", icon: improve },
    { name: "What is the Smooth Quant?", icon: improve },
    { name: "What is Neural Architecture Search?", icon: improve },
  ],
  "llama-7b": [
    {
      name: "If you could have any superpower, what would it be?",
      icon: improve,
    },
    { name: "Can you tell me a joke?", icon: improve },
    {
      name: "Can you recommend a good restaurant in Shanghai?",
      icon: improve,
    },
  ],
  "gpt-j-6b": [
    { name: "How do you define happiness?", icon: improve },
    { name: "What are your capabilities?", icon: improve },
    {
      name: "If you could travel anywhere in the world, where would you go and why?",
      icon: improve,
    },
  ],
  Customized: [],
};
export const MODEL_OPTION = {
  "names": ["llama-7b", "gpt-j-6b"],
  "options": [
    {
      label: "Max Tokens",
      value: 512,
      minRange: 0,
      maxRange: 1024,
      step: 1,
      type: "range",
    },
    {
      label: "Temperature",
      value: 0.1,
      minRange: 0,
      maxRange: 1.0,
      step: 0.1,
      type: "range",
    },
    {
      label: "Top P",
      value: 0.75,
      minRange: 0,
      maxRange: 1.0,
      step: 0.1,
      type: "range",
    },
    {
      label: "Top K",
      value: 1,
      minRange: 0,
      maxRange: 200,
      step: 1,
      type: "range",
    },
  ]
};
export const KNOWLEDGE_OPTIONS = ["Wikipedia", "INC Document", "Customized"]

export let DOMAIN_LIST = [
  {
    title: "Transport",
    style: "border-white",
    svg: transport,
    imgPosition: "stock-position",
  },
  {
    title: "Financial",
    style: "border-white",
    svg: financial,
    imgPosition: "education-position",
  },
  {
    title: "Stock",
    style: "border-white",
    svg: stock,
    imgPosition: "stock-position",
  },
  {
    title: "Education",
    style: "border-white",
    svg: education,
    imgPosition: "education-position",
  },
  {
    title: "Real Estate",
    style: "border-indigo-800",
    svg: realEstate,
    imgPosition: "real-position",
  },
];

