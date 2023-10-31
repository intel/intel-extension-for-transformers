import MavisBarryAvatar from "$lib/components/talkbot/imgs/Mavis-Barry-Avatar.jpg";
import LawrenceAvatar from "$lib/components/talkbot/imgs/Lawrence-Avatar.jpg";
import RichardJosephAvatar from "$lib/components/talkbot/imgs/Richard-Joseph-Avatar.jpg";
import TishaNortonAvatar from "$lib/components/talkbot/imgs/Tisha-Norton-Avatar.jpg";
import EmmaBergerAvatar from "$lib/components/talkbot/imgs/Emma-Berger-Avatar.jpg";
import EricYatesAvatar from "$lib/components/talkbot/imgs/Eric-Yates-Avatar.jpg";
import ShylaBandhuAvatar from "$lib/components/talkbot/imgs/Shyla-Bandhu-Avatar.jpg";

import indiaYouthVoice from "./assets/india.mp3"
import midAgeMantVoice from "./assets/mid-age-man.mp3"
import HumaVoice from "./assets/welcome_huma.wav"
import PatVoice from "./assets/mid-age-man.mp3"
import WeiVoice from "./assets/welcome_wei.wav"

import humaAvatar from "$lib/components/talkbot/imgs/huma.jpg";
import JohnAvatar from "$lib/components/talkbot/imgs/John-Avatar.jpg";
import annAvatar from "$lib/components/talkbot/imgs/wei.jpg";


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

]

export const TalkingKnowledgeLibrary = [
    { name: 'Wikipedia', },
    { name: 'Intel Neural Compressor', },
]

export const TalkingTemplateLibrary = [
    { name: "", avatar: JohnAvatar, audio: PatVoice, knowledge: 'Young_Pat', identify: "default_male" },
    { name: "", avatar: annAvatar, audio: WeiVoice, knowledge: 'Young_Wei', identify: "wei" },
    { name: "", avatar: humaAvatar, audio: HumaVoice, knowledge: 'Young_Huma', identify: "huma" },
    // { name: "Andy Grove", avatar: andyAvatar, audio: AndyVoice, knowledge: 'Young_Andy', identify: "andy" },
];

export const tabList = [
    { type: undefined, title: "Start a Talking Chat", },
    { type: "template", title: "Start with a Template", },
    { type: "customize", title: "Start with Customization", },
];