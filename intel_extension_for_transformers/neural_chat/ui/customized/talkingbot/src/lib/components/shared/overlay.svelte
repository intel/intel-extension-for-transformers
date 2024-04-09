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
	import { open } from './store';
</script>

<!-- The overlay will only be visible on small screens to emphasize the focus on the side navigation when it is open.-->
<div
	class={$open
		? 'bg-[#161730] fixed h-screen left-0 opacity-40 top-0 w-screen z-30 lg:bg-transparent'
		: ''}
/>
