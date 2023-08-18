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
		env: ({
			publicPrefix: ''
		})
	}
};

export default config;
