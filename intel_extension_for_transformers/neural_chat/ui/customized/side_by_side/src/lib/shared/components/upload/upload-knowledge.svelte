<!--
  Copyright (c) 2024 Intel Corporation
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

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
		class="focus:border-blue-700 focus:ring-0"
	  />
	</Label>
  </div>
  