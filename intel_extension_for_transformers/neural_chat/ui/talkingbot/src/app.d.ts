// See: https://kit.svelte.dev/docs/types#app
// import { Result} from "neverthrow";

declare namespace App {
	interface Locals {
		user?: User;
	}
}

interface ChatAdapter {
	modelList(props: {
	}): Promise<Result<void>>;
	txt2img(props: {
	}): Promise<Result<void>>;
}

interface ChatMessage {
	role: string,
	content: string
}
