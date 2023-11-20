<script>
	import 'tailwindcss/tailwind.css';
	import Layout from '$lib/components/layout.svelte';
	import Notifications from 'svelte-notifications';
</script>

<Layout>
	<Notifications>
		<slot />
	</Notifications>
</Layout>