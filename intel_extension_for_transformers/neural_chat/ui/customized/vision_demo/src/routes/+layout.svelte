<script>
	import "tailwindcss/tailwind.css";
	import "../app.postcss";
	import Notifications from "svelte-notifications";
	import Header from "$lib/modules/frame/header.svelte";
	import Footer from "$lib/modules/frame/footer.svelte";
  </script>
  
  <style>
	.custom-background {
	  background-image: url("$lib/assets/png/ChatBot_Background.png");
	  background-size: cover;
	  background-position: center;
	}
  </style>
  
  <Notifications>
	<div class="flex h-full flex-col custom-background">
	  <Header />
	  <slot />
	  <Footer />
	</div>
  </Notifications>
  