<script lang="ts">
  export let overrideClasses = '';

  const classes = overrideClasses ? overrideClasses : `w-6 h-6`;
</script>

<svg
  class={classes}
  xmlns="http://www.w3.org/2000/svg"
  fill="none"
  viewBox="0 0 24 24"
  stroke-width="1.5"
  stroke="currentColor"
>
  <path
    stroke-linecap="round"
    stroke-linejoin="round"
    d="M6 18L18 6M6 6l12 12"
  />
</svg>
