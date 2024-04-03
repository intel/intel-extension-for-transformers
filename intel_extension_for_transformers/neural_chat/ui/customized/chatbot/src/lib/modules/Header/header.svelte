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

    import IntelLogo from "$lib/assets/header/intelLogo.svelte";
  
  </script>
  <header
    class="sticky top-0 z-40 flex-none w-full mx-auto bg-[#0054ae] border-b border-gray-200 dark:border-gray-600 dark:bg-gray-800"
  >
    <nav
      class="bg-[#0054ae] dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-700 divide-gray-200 dark:divide-gray-700 px-2 sm:px-4 w-full py-1.5"
    >
      <div class="mx-auto flex flex-wrap justify-between items-center w-full">
          <IntelLogo />
        <span
          class="self-center whitespace-nowrap text-2xl font-semibold text-white ml-4"
          data-svelte-h="svelte-1hbktnk">ChatBot</span
        >
        <!-- <div class="w-full ms-auto lg:block lg:w-auto order-1 lg:order-none">
          <ul
            class="flex flex-col py-3 my-4 lg:flex-row lg:my-0 text-sm font-medium text-gray-900 dark:text-gray-300 gap-4"
          >
            <li>
              <a
                href="/"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                >Home</a
              >
            </li>
            <li>
              <a
                href="/docs/pages/introduction"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                >Docs</a
              >
            </li>
            <li>
              <a
                href="/docs/components/accordion"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                >Components</a
              >
            </li>
            <li>
              <a
                href="https://flowbite-svelte-icons.vercel.app/"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                >Icons</a
              >
            </li>
            <li>
              <a
                href="https://flowbite-svelte-blocks.vercel.app/"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                >Blocks</a
              >
            </li>
            <li>
              <a
                href="https://flowbite-svelte-admin-dashboard.vercel.app/"
                class="block py-2 pe-4 ps-3 md:p-0 rounded md:border-0 md:!ps-3 md:!py-2 lg:!ps-0 text-gray-700 hover:bg-gray-100 lg:hover:bg-transparent lg:border-0 lg:hover:text-primary-700 dark:text-gray-400 lg:dark:text-white lg:dark:hover:text-primary-700 dark:hover:bg-gray-700 dark:hover:text-white lg:dark:hover:bg-transparent lg:px-2 lg:mb-0"
                target="_blank">Dashboard</a
              >
            </li>
          </ul>
        </div>
        <div class="flex items-center ms-auto">
          <a
            href="https://github.com/themesberg/flowbite-svelte"
            class="focus:outline-none whitespace-normal m-0.5 rounded-lg p-2.5 focus:ring-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 hidden sm:inline-block dark:hover:text-white hover:text-gray-900 focus:ring-0"
            aria-label="View on GitHub"
            target="_blank"
            rel="noreferrer"
            ><span class="sr-only">View on GitHub</span>
            <svg
              class="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
              focusable="false"
              data-icon="github"
              role="img"
              ><path
                fill-rule="evenodd"
                d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                clip-rule="evenodd"
              ></path></svg
            ></a
          >
          <a
            href="https://discord.gg/4eeurUVvTy"
            class="focus:outline-none whitespace-normal m-0.5 rounded-lg p-2.5 focus:ring-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 hidden xl:inline-block dark:hover:text-white hover:text-gray-900 focus:ring-0"
            aria-label="Join community on Discord"
            target="_blank"
            rel="noreferrer"
            ><span class="sr-only">Join community on Discord</span>
            <svg
              class="w-6 h-6"
              aria-hidden="true"
              focusable="false"
              data-prefix="fab"
              data-icon="discord"
              role="img"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 640 512"
              ><path
                fill="currentColor"
                d="M524.5 69.84a1.5 1.5 0 0 0 -.764-.7A485.1 485.1 0 0 0 404.1 32.03a1.816 1.816 0 0 0 -1.923 .91 337.5 337.5 0 0 0 -14.9 30.6 447.8 447.8 0 0 0 -134.4 0 309.5 309.5 0 0 0 -15.14-30.6 1.89 1.89 0 0 0 -1.924-.91A483.7 483.7 0 0 0 116.1 69.14a1.712 1.712 0 0 0 -.788 .676C39.07 183.7 18.19 294.7 28.43 404.4a2.016 2.016 0 0 0 .765 1.375A487.7 487.7 0 0 0 176 479.9a1.9 1.9 0 0 0 2.063-.676A348.2 348.2 0 0 0 208.1 430.4a1.86 1.86 0 0 0 -1.019-2.588 321.2 321.2 0 0 1 -45.87-21.85 1.885 1.885 0 0 1 -.185-3.126c3.082-2.309 6.166-4.711 9.109-7.137a1.819 1.819 0 0 1 1.9-.256c96.23 43.92 200.4 43.92 295.5 0a1.812 1.812 0 0 1 1.924 .233c2.944 2.426 6.027 4.851 9.132 7.16a1.884 1.884 0 0 1 -.162 3.126 301.4 301.4 0 0 1 -45.89 21.83 1.875 1.875 0 0 0 -1 2.611 391.1 391.1 0 0 0 30.01 48.81 1.864 1.864 0 0 0 2.063 .7A486 486 0 0 0 610.7 405.7a1.882 1.882 0 0 0 .765-1.352C623.7 277.6 590.9 167.5 524.5 69.84zM222.5 337.6c-28.97 0-52.84-26.59-52.84-59.24S193.1 219.1 222.5 219.1c29.67 0 53.31 26.82 52.84 59.24C275.3 310.1 251.9 337.6 222.5 337.6zm195.4 0c-28.97 0-52.84-26.59-52.84-59.24S388.4 219.1 417.9 219.1c29.67 0 53.31 26.82 52.84 59.24C470.7 310.1 447.5 337.6 417.9 337.6z"
              ></path></svg
            ></a
          >
          <a
            href="https://www.youtube.com/channel/UC_Ms4V2kYDsh7F_CSsHyQ6A"
            class="focus:outline-none whitespace-normal m-0.5 rounded-lg p-2.5 focus:ring-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 hidden xl:inline-block dark:hover:text-white hover:text-gray-900 focus:ring-0"
            aria-label="Subscribe to YouTube channel"
            target="_blank"
            rel="noreferrer"
            ><span class="sr-only">Subscribe to YouTube channel</span>
            <svg
              class="w-6 h-6"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 576 512"
              ><path
                fill="currentColor"
                d="M549.655 124.083c-6.281-23.65-24.787-42.276-48.284-48.597C458.781 64 288 64 288 64S117.22 64 74.629 75.486c-23.497 6.322-42.003 24.947-48.284 48.597-11.412 42.867-11.412 132.305-11.412 132.305s0 89.438 11.412 132.305c6.281 23.65 24.787 41.5 48.284 47.821C117.22 448 288 448 288 448s170.78 0 213.371-11.486c23.497-6.321 42.003-24.171 48.284-47.821 11.412-42.867 11.412-132.305 11.412-132.305s0-89.438-11.412-132.305zm-317.51 213.508V175.185l142.739 81.205-142.739 81.201z"
              ></path></svg
            ></a
          >
          <button
            aria-label="Dark mode"
            type="button"
            class="text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none rounded-lg text-sm p-2.5 inline-block dark:hover:text-white hover:text-gray-900"
            ><span class="hidden dark:block"
              ><svg
                class="w-6 h-6"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
                ><path
                  d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1
    0 100-2H3a1 1 0 000 2h1z"
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                ></path></svg
              ></span
            >
            <span class="block dark:hidden"
              ><svg
                class="w-6 h-6"
                fill="currentColor"
                viewBox="0 0 20 20"
                xmlns="http://www.w3.org/2000/svg"
                ><path
                  d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"
                ></path></svg
              ></span
            ></button
          >
        </div>
        <a
          href="https://www.npmjs.com/package/flowbite-svelte"
          class="hidden sm:block"
          ><div
            class="divide-gray-200 dark:divide-gray-700 font-medium inline-flex items-center justify-center px-2.5 py-0.5 text-sm rounded bg-primary-100 text-primary-700 dark:bg-gray-700 dark:text-primary-700 border-primary-700 dark:border-primary-700 ms-2 xl:ms-6 hover:bg-primary-600 hover:text-white dark:hover:bg-primary-800 dark:hover:text-white"
          >
            v0.44.24
          </div></a
        >
        <button
          type="button"
          class="focus:outline-none whitespace-normal rounded-lg focus:ring-2 p-1.5 focus:ring-gray-400 hover:bg-gray-100 dark:hover:bg-gray-600 m-0 md:block lg:hidden hidden"
          aria-label="Open main menu"
          ><span class="sr-only">Open main menu</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            role="button"
            tabindex="0"
            width="24"
            height="24"
            class="h-6 w-6 shrink-0"
            aria-label="bars 3"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="2"
            ><path
              stroke="currentColor"
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M4 6h16M4 12h16M4 18h16"
            ></path>
          </svg></button
        > -->
      </div>
    </nav>
  </header>