<script lang="ts">
	import { Fileupload, Label } from "flowbite-svelte";
	import { createEventDispatcher } from "svelte";
  
	const dispatch = createEventDispatcher();
	let value;
  
	function handleInput(event: Event) {
	  const file = (event.target as HTMLInputElement).files![0];
  
	  if (!file) return;
  
	  const reader = new FileReader();
	  reader.onloadend = () => {
		if (!reader.result) return;
		const src = reader.result.toString();
		dispatch("upload", { src: src, fileName: file.name });
	  };
	  reader.readAsDataURL(file);
	}
  </script>
  
  <div>
	<Label class="space-y-2 mb-2">
	  <Fileupload
		bind:value
		on:change={handleInput}
		class="focus:border-blue-700 foucs:ring-0"
	  />
	</Label>
  </div>
  