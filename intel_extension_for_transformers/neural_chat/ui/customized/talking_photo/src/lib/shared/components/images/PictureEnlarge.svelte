<script lang="ts">
	import Add from "$lib/assets/image-info/svelte/Add.svelte";
	import EnlargeImage from "$lib/assets/image-info/svelte/EnlargeImage.svelte";
	import Minus from "$lib/assets/image-info/svelte/Minus.svelte";
	import RotateLeft from "$lib/assets/image-info/svelte/RotateLeft.svelte";
	import RotateRight from "$lib/assets/image-info/svelte/RotateRight.svelte";
	import { Button, Modal } from "flowbite-svelte";
	import { Icon } from "flowbite-svelte-icons";
	import { onMount } from "svelte";

	export let imgSrc: string;
	export let extraClass: string = "";
	export let enlargeClass: string = "";


	let imgEl: HTMLImageElement;
	let openModal: boolean;
	let rotationStyle = "0";
	let scale = 1;
	let rotationList = ["0", "45", "90", "135", "180", "225", "270", "315"];
	let currentRotation = 0;

	let translateV = { x: 0, y: 0 }
	let startPoint = { x: 0, y: 0 }
	let isTouching = false

	// let scaleV = { x1: 0, y1: 0, x2: 0, y2: 0}
	// let isScaling = false

	function changeImgClass(imgChange: string) {
		if (imgChange === "rotateLeft") {
			currentRotation = (currentRotation + 1) % rotationList.length;
			rotationStyle = `-${rotationList[currentRotation]}`;
		}
		if (imgChange === "rotateRight") {
			currentRotation = (currentRotation + 1) % rotationList.length;
			rotationStyle = `${rotationList[currentRotation]}`;
		}
		if (imgChange === "add") {
			scale += 0.1;
		}
		if (imgChange === "minus") {
			scale -= 0.1;
		}
	}

	function handlePointerDown(e: PointerEvent) {
		
		e.preventDefault()
		isTouching = true
		startPoint = { x: e.clientX, y: e.clientY }
	}

	function handlePointerUp(e: PointerEvent) {
		isTouching = false
		
	}

	function handlePointerMove(e: PointerEvent) {
		
		if (isTouching) {
			translateV = {
				x: e.clientX - startPoint.x,
				y: e.clientY - startPoint.y,
			}
			
		}
	}

	function handleTouchStart(e: TouchEvent) {
		// e.preventDefault();
		
		// let touches = e.touches;
		// let events = touches[0];
		// let events2 = touches[1];

		// scaleV.x1 = events.clientX;
		// scaleV.y1 = events.clientY;

		// isScaling = true;

		// if (events2) {
		// 	scaleV.x2 = events2.clientX;
		// 	scaleV.y2 = events2.clientY;
		// }
	}

	function handleTouchMove(e: TouchEvent) {
		// e.preventDefault();

		// if (isScaling) {
		// 	return;
		// }
		// let touches = e.touches;
		// let events = touches[0];
		// let events2 = touches[1];		

		// if (events2) {
		// 	if (scaleV.x2 === 0) scaleV.x2 = events2.clientX;
		// 	if (scaleV.y2 === 0) scaleV.y2 = events2.clientY;

		// 	const getDistance = (v) => {
		// 		return Math.hypot(v.x2 - v.x1, v.y2 - v.y1);
		// 	};

		// 	let zoom = getDistance({ x1: events.clientX, y1: events.clientY, x2: events2.clientX, y2: events2.clientY})
		// 		 		/ getDistance(scaleV);

		// 	scale = Math.min(scale * zoom, 3);
		// }
	}

	function handleTouchEnd(e: TouchEvent) {
		// isScaling = false
	}

	async function downloadImage() {
		openModal = true;
		const response = await fetch(imgSrc);
		const blob = await response.blob();

		const blobUrl = URL.createObjectURL(blob);

		const a = document.createElement("a");
		a.style.display = "none";
		a.href = blobUrl;
		a.download = "image.jpg";
		document.body.appendChild(a);
		a.click();

		window.URL.revokeObjectURL(blobUrl);
		document.body.removeChild(a);
	}
</script>

<Modal title="Current Image" bind:open={openModal} outsideclose
	on:close={() => { isTouching = false }}
>
	<div class="relative h-[24rem]">
		<img
			bind:this={imgEl}
			src={imgSrc}
			style="transform: translate({translateV.x}px, {translateV.y}px) scale({scale.toString()}) rotate({rotationStyle}deg);"
			alt=""
			class="m-auto h-[20rem] object-cover touch-none"
			on:pointerdown={handlePointerDown}
			on:pointerup={handlePointerUp}
			on:pointermove={handlePointerMove}
			on:touchstart={handleTouchStart}
			on:touchmove={handleTouchMove}
			on:touchend={handleTouchEnd}
			on:touchcancel={handleTouchEnd}
		/>
	</div>
	<div
			class="absolute max-sm:bottom-6 bottom-4 left-1/2 z-50 h-16 w-full max-w-lg -translate-x-1/2 rounded-full border border-gray-200 bg-[#fff] shadow "
		>
			<div class="mx-auto grid h-full max-w-lg grid-cols-5">
				<button
					on:click={() => {
						changeImgClass("rotateLeft");
					}}
					type="button"
					class="group inline-flex flex-col items-center justify-center rounded-l-full px-5 hover:bg-gray-100 active:border"
				>
					<RotateLeft />
				</button>

				<button
					on:click={() => {
						changeImgClass("rotateRight");
					}}
					type="button"
					class="group inline-flex flex-col items-center justify-center px-5 hover:bg-gray-100 active:border"
				>
					<RotateRight />
				</button>

				<div class="flex items-center justify-center">
					<button
						on:click={downloadImage}
						type="button"
						class="group inline-flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 font-medium hover:bg-[#3369FF] active:border"
					>
						<Icon
							name="download-solid"
							class="h-4 w-4 border-none text-white active:border-none"
						/>
					</button>
				</div>

				<button
					on:click={() => {
						changeImgClass("add");
					}}
					type="button"
					class="group inline-flex flex-col items-center justify-center px-5 hover:bg-gray-100 active:border"
				>
					<Add />
				</button>

				<button
					on:click={() => {
						changeImgClass("minus");
					}}
					type="button"
					class="group inline-flex flex-col items-center justify-center rounded-r-full px-5 hover:bg-gray-100 active:border"
				>
					<Minus />
				</button>
			</div>
		</div>
</Modal>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div
	on:click={() => {
		openModal = true;
	}}
	class="absolute  bg-[rgba(0,0,0,.4)] {extraClass}"
>
	<EnlargeImage {enlargeClass}/>
</div>
