<script lang="ts">
    export let hintClass: String;
    export let hintIcon: String;
    export let hintText: String ;
</script>

<div class="absolute border-[#e1e1e1] px-2 py-1.5 rounded {hintClass}">
    <div class="flex flex-row items-center ">
        <div class="mr-2">
            <div class="relative flex h-3 w-3 justify-center items-center">
                <span class="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-75"></span>
                <span class="inline-flex bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-violet-500 text-sm font-bold ">{hintIcon}</span>
            </div>
            
        </div>
        <div    >
            <span class="bg-clip-text text-transparent  bg-gradient-to-r from-blue-500 to-violet-500 text-sm font-bold">{hintText}</span>
        </div>
    </div>
</div>