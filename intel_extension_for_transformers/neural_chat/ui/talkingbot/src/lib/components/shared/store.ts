import { writable } from 'svelte/store';

let open = writable(false);

export { open };
