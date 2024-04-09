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
    export let running = false
</script>

{#if running}
    <div class="img bg-cover -mb-20"></div>
{/if}

<style>
    .img {
        width: 600px;
        height: 300px;
        animation-name: flowingAnimation;
        animation-duration: 3s;
        animation-iteration-count: infinite;
        /* animation-timing-function: cubic-bezier(0.1, 0.7, 1, 0.1); */
        animation-timing-function: linear;
    }

    @keyframes flowingAnimation {
        0% {
            background-image: url(./imgs/1.svg)
        }

        20% {
            background-image: url(./imgs/2.svg)
        }

        40% {
            background-image: url(./imgs/3.svg)
        }

        60% {
            background-image: url(./imgs/4.svg)
        }

        80% {
            background-image: url(./imgs/5.svg)
        }

        100% {
            background-image: url(./imgs/1.svg)
        }
    }
</style>