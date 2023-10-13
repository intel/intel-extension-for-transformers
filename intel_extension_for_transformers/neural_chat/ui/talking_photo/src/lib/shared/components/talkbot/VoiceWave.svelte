<script lang="ts">
    export let running = false
</script>

{#if running}
    <div class="img waveStyle"></div>
{/if}

<style>
    .img {
        width: 400px;
        height: 200px;
        animation-name: flowingAnimation;
        animation-duration: 3s;
        animation-iteration-count: infinite;
        /* animation-timing-function: cubic-bezier(0.1, 0.7, 1, 0.1); */
        animation-timing-function: linear;
    }

    @keyframes flowingAnimation {
        0% {
            background-image: url(../../../assets/chat/svg/Wave1.svg)
        }

        20% {
            background-image: url(../../../assets/chat/svg/Wave2.svg)
        }

        40% {
            background-image: url(../../../assets/chat/svg/Wave3.svg)
        }

        60% {
            background-image: url(../../../assets/chat/svg/Wave4.svg)
        }

        80% {
            background-image: url(../../../assets/chat/svg/Wave5.svg)
        }

        100% {
            background-image: url(../../../assets/chat/svg/Wave1.svg)
        }
    }

    .waveStyle {
		position: absolute;
		top: calc(50% - 230px);
		left: 50%;
		transform: translateX(-50%);
	}
</style>