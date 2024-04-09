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
	export let mode: string;

	const title_map: {[key:string]: string} = {
		basic: 'Used For?',
		advanced: 'Q&A',
		chatgpt: 'ChatGPT',
	}
</script>

<div
	class="font-title-intel text-xl text-white mb-4 pb-4 border-b flex gap-4 overflow-hidden"
>
	<span class="text-lg font-bold">{title_map[mode]}</span>
</div>
