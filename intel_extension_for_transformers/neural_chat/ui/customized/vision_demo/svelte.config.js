// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import adapter from '@sveltejs/adapter-auto';
import preprocess from 'svelte-preprocess';
import postcssPresetEnv from 'postcss-preset-env';


/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://github.com/sveltejs/svelte-preprocess
	// for more information about preprocessors
	preprocess: preprocess({
		sourceMap: true,
		postcss: {
			plugins: [postcssPresetEnv({ features: { 'nesting-rules': true } })]
		}
	}),

	kit: {
		adapter: adapter(),
		env: {
			publicPrefix: ''
		}
	}
};

export default config;
