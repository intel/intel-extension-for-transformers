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
