import HomeIcon from './icons/HomeIcon.svelte';
import DocumentationIcon from './icons/DocumentationIcon.svelte';
import AvatarIcon from './icons/AvatarIcon.svelte'
import TemplateIcon from './icons/TemplateIcon.svelte'
import VoiceIcon from './icons/VoiceIcon.svelte'
import KnowledgeIcon from './icons/KnowledgeIcon.svelte'
interface MenuItem {
  title: string;
  icon: typeof import('*.svelte').default;
  link: string;
}

const data: MenuItem[] = [
  {
    title: 'Home',
    icon: DocumentationIcon,
    link: '/'
  }, {
    title: 'Template',
    icon: TemplateIcon,
    link: '/template'
  }, {
    title: 'Avatar',
    icon: AvatarIcon,
    link: '/avatar'
  }, {
    title: 'Voice',
    icon: VoiceIcon,
    link: '/voice'
  }, {
    title: 'Knowledge',
    icon: KnowledgeIcon,
    link: '/knowledge'
  }, {
    title: 'Chat',
    icon: HomeIcon,
    link: '/chat'
  },
];

export default data;
