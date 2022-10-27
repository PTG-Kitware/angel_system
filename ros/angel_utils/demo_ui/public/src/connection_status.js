// connection status
ros.on('connection', function() {
    document.getElementById("status-msg").innerHTML = "Connected";
});

ros.on('error', function(error) {
    document.getElementById("status-msg").innerHTML = "Error: (${error})";
});

ros.on('close', function() {
    document.getElementById("status-msg").innerHTML = "Closed";
});
  