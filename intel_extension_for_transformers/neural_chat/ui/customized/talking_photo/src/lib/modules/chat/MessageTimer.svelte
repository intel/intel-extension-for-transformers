<script lang="ts">
	export let time: string;
	// const messageTime = new Date();
	// let messageTimer = "Just now";

	// setInterval(function () {
	// 	var now = new Date(),
	// 		current = now.getTime();

	// 	messageTimer = timeDifference(current, messageTime.getTime());
	// }, 5000);

	// function timeDifference(current: number, previous: number) {
	// 	const msPerMinute = 60 * 1000;
	// 	const msPerHour = msPerMinute * 60;
	// 	const msPerDay = msPerHour * 24;
	// 	const msPerMonth = msPerDay * 30;
	// 	const msPerYear = msPerDay * 365;

	// 	const elapsed = current - previous;

	// 	if (elapsed < msPerMinute) {
	// 		return Math.round(elapsed / 1000) + ' seconds ago';
	// 	} else if (elapsed < msPerHour) {
	// 		return Math.round(elapsed / msPerMinute) + ' minutes ago';
	// 	} else if (elapsed < msPerDay) {
	// 		return Math.round(elapsed / msPerHour) + ' hours ago';
	// 	} else if (elapsed < msPerMonth) {
	// 		return 'approximately ' + Math.round(elapsed / msPerDay) + ' days ago';
	// 	} else if (elapsed < msPerYear) {
	// 		return 'approximately ' + Math.round(elapsed / msPerMonth) + ' months ago';
	// 	} else {
	// 		return 'approximately ' + Math.round(elapsed / msPerYear) + ' years ago';
	// 	}
	// }
</script>

<!-- <span class="text-xs text-gray-500 leading-none">{messageTimer}</span> -->
<p class="text-xs text-center opacity-70">
	{time}
</p>