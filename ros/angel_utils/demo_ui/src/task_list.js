// Get data from angel_system
workspace = '/debug';
var ros = new ROSLIB.Ros({
  url : 'ws://localhost:9090'
});

// subscribe to QueryTaskGraph
var query_task_graph = new ROSLIB.Service({
  ros: ros,
  name: workspace + '/query_task_graph',
  serviceType: 'angel_msgs/srv/QueryTaskGraph'
});

var request = {};
var user_level = 1;
query_task_graph.callService(request, function(result){
  // Load title
  var task_title = result.task_title;
  var title_container_block = document.getElementById('task_title');
  title_container_block.innerHTML = task_title;

  // Load tasks
  var task_list = result.task_graph.task_steps;
  var task_levels = result.task_graph.task_levels;
  var container_block = document.getElementById('task-list');

  task_list.forEach(function(task, index){
    var task_level = task_levels[index];
    if(task_level == user_level){
      var task_line = document.createElement('div');
      task_line.className = "task-line";

      var checkbox = document.createElement('span');
      checkbox.className = "checkbox";
      checkbox.id = task;
      task_line.appendChild(checkbox);

      var text = document.createElement('span');
      text.className = "text body-text task";
      text.innerHTML = task;
      task_line.appendChild(text);

      container_block.appendChild(task_line);
    }
  });
});

// Create a listener for task completion updates
var task_update = new ROSLIB.Topic({
  ros : ros,
  name : workspace + '/TaskUpdates',
  messageType : 'angel_msgs/msg/TaskUpdate' // find: $ ros2 topic type <>
});

task_update.subscribe(function(m) {
  var task_name = m.previous_step;
  var empty_checkbox = document.getElementById(task_name);

  if(empty_checkbox != null){
    var checkmark = document.createElement('span');
    checkmark.className = "checkmark";
    empty_checkbox.appendChild(checkmark);
  }
});

// Done button
function done() {
  console.log("Done!");
}

var done_btn = document.getElementById("done-btn");
done_btn.onclick = done;
