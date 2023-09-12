<script lang="ts">
    export let knowledge_question:string

    let showSearch = false
    let showGenerate = false

    setTimeout(() => showSearch= true , 3000);
    setTimeout(() => showGenerate= true , 10000);

</script>

{#if showSearch}
    <div class="flex items-center gap-2 my-2">
        <svg class="w-6 h-6" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg">
            <path d="M912 190h-69.9c-9.8 0-19.1 4.5-25.1 12.2L404.7 724.5 207 474c-6.1-7.7-15.3-12.2-25.1-12.2H112c-6.7 0-10.4 7.7-6.3 12.9l273.9 347c12.8 16.2 37.4 16.2 50.3 0l488.4-618.9c4.1-5.1 0.4-12.8-6.3-12.8z" fill="#59bd59"></path>
        </svg>
        <p>Searching for:  <span class="font-bold">{knowledge_question}</p>
    </div>
{/if}

{#if showGenerate}
    <div class="flex items-center gap-2 my-2">
        <svg class="w-6 h-6" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg">
            <path d="M912 190h-69.9c-9.8 0-19.1 4.5-25.1 12.2L404.7 724.5 207 474c-6.1-7.7-15.3-12.2-25.1-12.2H112c-6.7 0-10.4 7.7-6.3 12.9l273.9 347c12.8 16.2 37.4 16.2 50.3 0l488.4-618.9c4.1-5.1 0.4-12.8-6.3-12.8z" fill="#59bd59"></path>
        </svg>
        <p>Generating answers for you...</p>
    </div>
{/if}
