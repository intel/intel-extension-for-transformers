import { derived, writable } from 'svelte/store';

import type {
  Chat,
  ChatListItem,
} from './shared.type';

/**
 * Banners
 */
export const banners$ = writable([] as any);

export const hasBanners$ = derived(banners$, (banners) => {
  return banners.length > 0;
});

/**
 * localStorage
 */
export const chatList$ = writable([] as ChatListItem[]);
export const chats$ = writable({} as Record<string, Chat>);

