import Chat from '$lib/assets/frame/svelte/Chat.svelte';
import Info from '$lib/assets/frame/svelte/Info.svelte'
import Setting from '$lib/assets/frame/svelte/Setting.svelte'
import Time from '$lib/assets/image-info/svg/Time.svg'
import Location from '$lib/assets/image-info/svg/Location.svg'
import Character from '$lib/assets/image-info/svg/Character.svg'
import midAgeMantVoice from '$lib/assets/voice/audio/mid-age-man.mp3'
import indiaYouthVoice from '$lib/assets/voice/audio/scottish_women.mp3'

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
    title: 'Photo',
    icon: Info,
    link: '/info'
  },
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

export const TalkingVoiceLibrary = [
  {name: 'Indian Youth Woman', audio: indiaYouthVoice},
  {name: 'Middle-Aged Man', audio: midAgeMantVoice},
]