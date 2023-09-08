import MavisBarryAvatar from "$lib/components/talkbot/imgs/Mavis-Barry-Avatar.png";
import LawrenceAvatar from "$lib/components/talkbot/imgs/Lawrence-Avatar.png";
import RichardJosephAvatar from "$lib/components/talkbot/imgs/Richard-Joseph-Avatar.png";
import TishaNortonAvatar from "$lib/components/talkbot/imgs/Tisha-Norton-Avatar.png";
import EmmaBergerAvatar from "$lib/components/talkbot/imgs/Emma-Berger-Avatar.png";
import EricYatesAvatar from "$lib/components/talkbot/imgs/Eric-Yates-Avatar.png";
import ShylaBandhuAvatar from "$lib/components/talkbot/imgs/Shyla-Bandhu-Avatar.png";

import indiaYouthVoice from "./assets/india.mp3"
import midAgeMantVoice from "./assets/mid-age-man.mp3"
import robotVoice from "./assets/bot.mp3"
import childVoice from "./assets/child.mp3"
import scotlandYouthVoice from "./assets/scottish_young.mp3"
import scotlandWomanVoice from "./assets/scottish_women.mp3"
import HumaVoice from "./assets/welcome_huma.wav"
import PatVoice from "./assets/welcome_pat.wav"
import AndyVoice from "./assets/welcome_andy.wav"
import WeiVoice from "./assets/welcome_wei.wav"

// import humaAvatar from "$lib/components/talkbot/imgs/huma-Avatar.png";
import humaAvatar from "$lib/components/talkbot/imgs/huma.png";
import JohnAvatar from "$lib/components/talkbot/imgs/John-Avatar.png";
import annAvatar from "$lib/components/talkbot/imgs/wei.png";
import andyAvatar from "$lib/components/talkbot/imgs/Andy-Avatar.png";


export const TalkingPhotoLibrary = [
    { name: 'Mavis Barry', avatar: MavisBarryAvatar },
    { name: 'Lawrence', avatar: LawrenceAvatar },
    { name: 'Richard Joseph', avatar: RichardJosephAvatar },
    { name: 'Tisha Norton', avatar: TishaNortonAvatar },
    { name: 'Emma Berger', avatar: EmmaBergerAvatar },
    { name: 'Eric Yates', avatar: EricYatesAvatar },
    { name: 'Shyla Bandhu', avatar: ShylaBandhuAvatar },
]

export const TalkingVoiceLibrary = [
    { name: 'Woman', audio: indiaYouthVoice, identify: "default" },
    { name: 'Man', audio: midAgeMantVoice, identify: "default_male" },
    // { name: 'Robot', audio: robotVoice },
    // { name: 'Child', audio: childVoice },
    // { name: 'Scotland Youth', audio: scotlandYouthVoice },
    // { name: 'Scotland Woman', audio: scotlandWomanVoice },
]

export const TalkingKnowledgeLibrary = [
    { name: 'Wikipedia', },
    { name: 'Intel Neural Compressor', },
    // { name: 'Young_Pat', },
]

export const TalkingTemplateLibrary = [
    { name: "Pat Gelsinger", avatar: JohnAvatar, audio: PatVoice, knowledge: 'Young_Pat', identify: "pat" },
    { name: "Wei Li", avatar: annAvatar, audio: WeiVoice, knowledge: 'Young_Wei', identify: "wei" },
    { name: "Huma Abidi", avatar: humaAvatar, audio: HumaVoice, knowledge: 'Young_Huma', identify: "huma" },
    { name: "Andy Grove", avatar: andyAvatar, audio: AndyVoice, knowledge: 'Young_Andy', identify: "andy" },
];

export const tabList = [
    { type: undefined, title: "Start a Talking Chat", },
    { type: "template", title: "Start with a Template", },
    { type: "customize", title: "Start with Customization", },
];