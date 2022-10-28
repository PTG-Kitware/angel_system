// Create a listener for error notifications
$.get( "/ns")
.done(function( data ){
    var error_listener = new ROSLIB.Topic({
        ros : ros,
        name : data.namespace + '/TaskErrors',
        messageType : 'angel_msgs/msg/AruiUserNotification' // find: $ ros2 topic type <>
    });

    error_listener.subscribe(function(m) {
        // Report errors
        if (m.context === 0) {
            var ctx = document.getElementById("error-msgs");
            var container = document.createElement('div');
            container.className = "error-msg";

            var title = "TASK ERROR: " + m.title;

            var btn = document.createElement('button');
            btn.className = "collapsible lapse text body-text";
            btn.innerText = btn.textContent = title;

            var content_box = document.createElement('div');
            content_box.className = "content text body-text";

            var msg = document.createElement('p');
            msg.innerHTML = m.description;
            content_box.appendChild(msg);

            // Insert
            container.appendChild(btn);
            container.appendChild(content_box);

            ctx.insertBefore(container, ctx.firstChild);

            // Expand text when clicked
            btn.addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            }); 
        }
    });
});
