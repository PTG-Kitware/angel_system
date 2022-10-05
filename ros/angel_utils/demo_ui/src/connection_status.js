var ros = new ROSLIB.Ros({
    url : 'ws://localhost:9090'
});

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

// Create a listener for /my_topic
const my_topic_listener = new ROSLIB.TOPIC({
    ros,
    name: "/my_topic",
    messageType: "std_msgs/String",
});
  
// When we receive a message on /my_topic, add its data as a list item to the "messages" ul
my_topic_listener.subscribe((message) => {
    const ul = document.getElementById("messages");
    const newMessage = document.createElement("li");
    newMessage.appendChild(document.createTextNode(message.data));
    ul.appendChild(newMessage);
});
  