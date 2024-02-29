<script lang="ts">
	import { Drawer, Button, CloseButton, Tabs, TabItem } from "flowbite-svelte";
	import { InfoCircleSolid } from "flowbite-svelte-icons";
	import { sineIn } from "svelte/easing";
	import UploadFile from "./upload-knowledge.svelte";
	import PasteURL from "./PasteKnowledge.svelte";
	import {
		knowledge1,
		knowledge2,
		knowledgeName,
	} from "$lib/shared/stores/common/Store";
	import DeleteIcon from "$lib/assets/avatar/svelte/Delete.svelte";
	import { getNotificationsContext } from "svelte-notifications";
	import {
		fetchKnowledgeBaseId,
		fetchKnowledgeBaseId2,
		fetchKnowledgeBaseIdByPaste,
	} from "$lib/network/upload/Network";

	const { addNotification } = getNotificationsContext();
	console.log("allKnowledges", $knowledgeName);

	let hidden6 = true;
	let selectKnowledge = -1;
	let transitionParamsRight = {
		x: 320,
		duration: 200,
		easing: sineIn,
	};

	async function handleKnowledgePaste(
		e: CustomEvent<{ pasteUrlList: string[] }>
	) {
		let knowledge_id = "";
		// let knowledge_id2 = "";
		try {
			const pasteUrlList = e.detail.pasteUrlList;
			const res = await fetchKnowledgeBaseIdByPaste(pasteUrlList, "url1");
			const res2 = await fetchKnowledgeBaseIdByPaste(pasteUrlList, "url2");
			// sihan
			knowledge_id = res.knowledge_base_id ? res.knowledge_base_id : "default";
			// knowledge_id2 = res2.knowledge_base_id ? res2.knowledge_base_id : "default";
		} catch {
			knowledge_id = "default";
			// knowledge_id2 = "default";
		}
		knowledge1.set({ id: knowledge_id });
		knowledgeName.set('knowledge_base');
		// knowledge2.set({ id: knowledge_id2 })

		addNotification({
			text: "Uploaded successfully",
			position: "top-left",
			type: "success",
			removeAfter: 3000,
		});
	}

	async function handleKnowledgeUpload(e: CustomEvent<any>) {
		let knowledge_id = "";
		// let knowledge_id2 = "";
		try {
			const blob = await fetch(e.detail.src).then((r) => r.blob());
			const fileName = e.detail.fileName;
			// letong
			const res = await fetchKnowledgeBaseId(blob, fileName);
			// const res2 = await fetchKnowledgeBaseId2(blob, fileName);
			// sihan
			knowledge_id = res.knowledge_base_id ? res.knowledge_base_id : "default";
			// knowledge_id2 = res2.knowledge_base_id ? res2.knowledge_base_id : "default";
			console.log("knowledge_id", knowledge_id);
		} catch {
			knowledge_id = "default";
			// knowledge_id2 = "default";
		}
		knowledge1.set({ id: knowledge_id });
		knowledgeName.set(e.detail.fileName);
		// knowledge2.set({ id: knowledge_id2 })
		addNotification({
			text: "Uploaded successfully",
			position: "top-left",
			type: "success",
			removeAfter: 3000,
		});
	}

	function handleKnowledgeDelete() {
		knowledge1.set({ id: "default" });
		knowledge2.set({ id: "default" });
		knowledgeName.set("");
	}
</script>

<div class="text-center">
	<Button
		on:click={() => (hidden6 = false)}
		class="bg-transparent focus-within:ring-gray-300 hover:bg-transparent focus:ring-0"
	>
		<svg
			aria-hidden="true"
			class="h-7 w-7 text-blue-700"
			fill="none"
			stroke="currentColor"
			viewBox="0 0 24 24"
			xmlns="http://www.w3.org/2000/svg"
			><path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
			/></svg
		>
	</Button>
</div>

<Drawer
	placement="right"
	transitionType="fly"
	transitionParams={transitionParamsRight}
	bind:hidden={hidden6}
	id="sidebar6"
>
	<div class="flex items-center">
		<h5
			id="drawer-label"
			class="mb-4 inline-flex items-center text-base font-semibold text-gray-500 dark:text-gray-400"
		>
			<InfoCircleSolid class="me-2.5 h-4 w-4" />Upload File
		</h5>
		<CloseButton
			on:click={() => (hidden6 = true)}
			class="mb-4 dark:text-white"
		/>
	</div>
	<p class="mb-6 text-sm text-gray-500 dark:text-gray-400">
		Please upload your local file or paste a remote file link, and Chat will
		respond based on the content of the uploaded file.
	</p>
	<Tabs
		style="full"
		defaultClass="flex rounded-lg divide-x rtl:divide-x-reverse divide-gray-200 shadow dark:divide-gray-700 foucs:ring-0"
	>
		<TabItem class="w-full" open>
			<span slot="title">Upload File</span>
			<UploadFile on:upload={handleKnowledgeUpload} />
		</TabItem>
		<TabItem class="w-full">
			<span slot="title">Paste Link</span>
			<PasteURL on:paste={handleKnowledgePaste} />
		</TabItem>
	</Tabs>
	{#if ($knowledgeName) && ($knowledgeName !== "")}
		<div class="relative">
			<p class="border-b p-6 pb-2">{$knowledgeName}</p>
			<DeleteIcon on:DeleteAvatar={() => handleKnowledgeDelete()} />
		</div>
	{/if}
</Drawer>
