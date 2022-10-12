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
  // Add checkmark to task
  var task_name = m.previous_step;
  var empty_checkbox = document.getElementById(task_name);

  if(empty_checkbox != null){
    var existing_check = empty_checkbox.querySelector("#checkmark");

    if(existing_check == null){
      // only add checkbox if it isn't there
      var checkmark = document.createElement('span');
      checkmark.className = "checkmark";
      checkmark.id = "checkmark";
      empty_checkbox.appendChild(checkmark);
    }
  }

  // Remove checkmark from next step
  var next_task_name = m.current_step;
  var next_checkbox = document.getElementById(next_task_name);

  if(next_checkbox != null){
    var next_existing_check = next_checkbox.querySelector("#checkmark");

    if(next_existing_check != null){
      next_checkbox.removeChild(next_existing_check);
    }
  }

  // Update colors in chart
  var chart = Chart.getChart('activity-conf');
  var colors = new Array(xValues.length).fill("rgba(0, 104, 199, 1.0)");
  var idx = chart.data.labels.indexOf(task_name);

  for(i=0; i<=idx; i++){
    colors[i] = "rgba(62, 174, 43, 1.0)"; // green
  }
  if(idx+1 < colors.length){
    colors[idx+1] = "rgb(254, 219, 101)"; // yellow
  }
  colors[chart.data.labels.indexOf("Background")] = "rgba(0, 104, 199, 1.0)"; // blue

  chart.data.datasets[0].backgroundColor = colors;

  chart.update('none');
});

// Done button
function done() {
  console.log("Done!");
}

var done_btn = document.getElementById("done-btn");
done_btn.onclick = done;
