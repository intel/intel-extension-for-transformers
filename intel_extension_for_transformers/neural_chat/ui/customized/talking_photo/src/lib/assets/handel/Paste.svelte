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

<script>
    import paste from '$lib/assets/voice/svg/paste.svg';

</script>

<!-- svelte-ignore a11y-label-has-associated-control -->
<label class="flex flex-col items-center justify-center w-full aspect-square shadow-[0_2px_30px_0_rgba(0,0,0,0.1)] rounded-xl text-center">
	<img class="h-7 swap-on" src={paste} alt=""/>
	<p class="text-xs text-gray-500 mt-1">URL</p>
</label>