import Chat from '$lib/assets/frame/svelte/Chat.svelte';
import Customize from '$lib/assets/frame/svelte/Customize.svelte'
import Info from '$lib/assets/frame/svelte/Info.svelte'
import VideoIcon from '$lib/assets/frame/svelte/Video.svelte'
import HintIcon from '$lib/assets/frame/svelte/Hint.svelte'
import Setting from '$lib/assets/frame/svelte/Setting.svelte'
import Time from '$lib/assets/image-info/svg/Time.svg'
import Location from '$lib/assets/image-info/svg/Location.svg'
import Character from '$lib/assets/image-info/svg/Character.svg'
import midAgeMantVoice from '$lib/assets/voice/audio/mid-age-man.mp3'
import indiaYouthVoice from '$lib/assets/voice/audio/scottish_women.mp3'

import MavisBarryAvatar from "$lib/assets/avatar/img/Mavis-Barry-Avatar.jpg";
import LawrenceAvatar from "$lib/assets/avatar/img/Lawrence-Avatar.jpg";
import RichardJosephAvatar from "$lib/assets/avatar/img/Richard-Joseph-Avatar.jpg";
import DeeparkAvatar from "$lib/assets/avatar/img/DeeparkAvatar-Avatar.png";
import TishaNortonAvatar from "$lib/assets/avatar/img/Tisha-Norton-Avatar.jpg";
import EmmaBergerAvatar from "$lib/assets/avatar/img/Emma-Berger-Avatar.jpg";
import EricYatesAvatar from "$lib/assets/avatar/img/Eric-Yates-Avatar.jpg";
import ShylaBandhuAvatar from "$lib/assets/avatar/img/Shyla-Bandhu-Avatar.jpg";

import HumaVoice from "$lib/assets/voice/audio/huma.wav"
import WeiVoice from "$lib/assets/voice/audio/li_wei.wav"
import DeepakVoice from "$lib/assets/voice/audio/deepak.wav"

import humaAvatar from "$lib/assets/customize/img/huma.jpg";
import JohnAvatar from "$lib/assets/customize/img/John-Avatar.jpg";
import annAvatar from "$lib/assets/customize/img/wei.jpg";

// sidebar ----------
interface MenuItem {
  title: string;
  icon: typeof import('*.svelte').default;
  link: string;
}

export const data: MenuItem[] = [
  {
    title: 'Chat',
    icon: Chat,
    link: '/'
  },
  {
    title: 'Customize',
    icon: Customize,
    link: '/customize'
  },
  // {
  //   title: 'Photo',
  //   icon: Info,
  //   link: '/classify'
  // },
  
  // {
  //   title: 'Voice',
  //   icon: Info,
  //   link: '/voice'
  // },
  {
    title: 'Settings',
    icon: Setting,
    link: '/settings'
  },

];
// sidebar ----------

export const COLORS: Array<
		| "green"
		| "yellow"
		| "indigo"
		| "purple"
		| "pink"
		| "none"
		| "blue"
		| "primary"
		| "red"
		| "dark"
		| undefined
	> = ["green", "yellow", "indigo", "purple", "pink", "red", "dark"];

export const imgTest = [
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
  { src: "https://img.zcool.cn/community/0131565aeff3c5a801219b7f6906a7.gif" },
]  

export const dummyImgList = [
  { src: "https://flowbite-svelte.com/images/components/carousel.svg", class: "mx-auto h-[4rem] w-[4rem] object-cover object-center max-sm:h-[3rem] max-sm:w-[3rem]" },
  { src: "https://flowbite-svelte.com/images/components/carousel.svg", class: "mx-auto h-[4rem] w-[4rem] object-cover object-center max-sm:h-[3rem] max-sm:w-[3rem]" },
  { src: "https://flowbite-svelte.com/images/components/carousel.svg", class: "mx-auto h-[4rem] w-[4rem] object-cover object-center max-sm:h-[3rem] max-sm:w-[3rem]" },
]

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
  { name: 'Man', audio: midAgeMantVoice, identify: "male" },
]

export const TalkingKnowledgeLibrary = [
  { name: 'Wikipedia', id: "default" },
  { name: 'Intel Neural Compressor', id: "default" },
]

export const TalkingTemplateLibrary = [
  { name: "Wei", avatar: annAvatar, audio: WeiVoice, identify: "wei", knowledge: "default", knowledge_name: "default" },
  { name: "Deepak", avatar: DeeparkAvatar, audio: DeepakVoice, identify: "deepak", knowledge: "default", knowledge_name: "default" },
  // { name: "Huma", avatar: humaAvatar, audio: HumaVoice, identify: "huma", knowledge: "default", knowledge_name: "default" },
];

export const toolList = [
  {
    title: 'Hint',
    icon: HintIcon,
    link: '/info'
  },
  {
    title: 'Video',
    icon: VideoIcon,
    link: '/info'
  },
]