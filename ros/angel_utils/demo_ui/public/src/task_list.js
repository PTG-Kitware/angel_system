$.get( "/ns")
.done(function( data ){
  // subscribe to QueryTaskGraph
  var query_task_graph = new ROSLIB.Service({
    ros: ros,
    name: data.namespace + '/query_task_graph',
    serviceType: 'angel_msgs/srv/QueryTaskGraph'
  });

  var request = {};
  var task_list;
  query_task_graph.callService(request, function(result){
    // Load title
    var task_title = result.task_title;
    var title_container_block = document.getElementById('task_title');
    title_container_block.innerHTML = task_title;

    // Load tasks
    task_list = result.task_graph.task_steps;
    var task_levels = result.task_graph.task_levels;
    var container_block = document.getElementById('task-list');

    task_list.forEach(function(task, index){
      // TODO: support different task levels
      var task_level = task_levels[index];

      var task_line = document.createElement('div');
      task_line.className = "task-line";

      var checkbox = document.createElement('span');
      checkbox.className = "checkbox";
      checkbox.id = task;
      task_line.appendChild(checkbox);

      var checkmark = document.createElement('span');
      checkmark.className = "checkmark_hidden checkmark";
      checkmark.id = "checkmark";
      checkbox.appendChild(checkmark);

      var text = document.createElement('span');
      text.className = "text body-text task";
      text.innerHTML = task;
      task_line.appendChild(text);

      container_block.appendChild(task_line);

    });
  });

  // Create a listener for task completion updates
  var task_update = new ROSLIB.Topic({
    ros : ros,
    name : data.namespace + '/TaskUpdates',
    messageType : 'angel_msgs/msg/TaskUpdate' // find: $ ros2 topic type <>
  });

  task_update.subscribe(function(m) {
    // Update checkmarks
    var task_name = m.current_step;
    var task_idx = m.current_step_id; // -1 at start

    task_list.forEach(function(task, index){
      var el = document.getElementById(task);

      if (index <= task_idx){
        // Add checkmark to all tasks up to and including task
        el.querySelector('.checkmark').className = 'checkmark_visible checkmark';
      }
      else{
        // Remove checkmarks for all tasks after the task
        el.querySelector('.checkmark').className = 'checkmark_hidden checkmark';
      }
    });
  });

  // Done button
  function done() {
    // TODO: Update this
    console.log("Done!");
  }

  var done_btn = document.getElementById("done-btn");
  done_btn.onclick = done;
});
