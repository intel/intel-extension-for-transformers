<script lang="ts">
    import timeDifference from '$lib/common/timediff';

    const messageTime = new Date()
    let messageTimer = 'Just now'

    setInterval(function() {
        var now = new Date(),
        current = now.getTime();
    
        messageTimer = timeDifference(current, messageTime.getTime())
    }, 5000);
</script>

<span class="text-xs text-gray-500 leading-none">{messageTimer}</span>
