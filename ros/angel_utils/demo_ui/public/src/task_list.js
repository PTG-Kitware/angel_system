var task_ctx = document.getElementById("task-complete-chart").getContext('2d');

var task_complete_chart = new Chart(task_ctx, {
  type: "bar",
  data: {
    labels: ["task completion"],
    datasets: [{
      backgroundColor: "rgba(0, 104, 199, 1.0)",
      data: [0]
    }]
  },
  options: {
    scales: {
        x: { display: false },
        y: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.1
          }
        }
    },
    title: {
      display: false,
    },
    maintainAspectRatio: false
  }
});

$.get( "/ns")
.done(function( data ){
  var colors;

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
      text.innerHTML = index+1 + '. ' + task;
      task_line.appendChild(text);

      container_block.appendChild(task_line);
    });

    colors = new Array(task_list.length+1).fill("rgba(0, 104, 199, 1.0)");
  });

  // Create a listener for task completion updates
  var task_update = new ROSLIB.Topic({
    ros : ros,
    name : data.namespace + '/TaskUpdates',
    messageType : 'angel_msgs/msg/TaskUpdate' // find: $ ros2 topic type <>
  });

  task_update.subscribe(function(m) {
    var task_name = m.current_step;
    var task_idx = m.current_step_id; // -1 at start
    var previous_name = m.previous_step;
    var previous_idx = task_list.indexOf(previous_name);

    if( previous_idx > task_idx ) {
      // We are going backwards, remove checks after current_step
      for(var i=task_idx+1; i<=previous_idx; i++) {
        var el = document.getElementById(task_list[i]);
        el.querySelector('.checkmark').className = 'checkmark_hidden checkmark';
      }
    }
    else {
      var el = document.getElementById(task_name);
      el.querySelector('.checkmark').className = 'checkmark_visible checkmark';
    }

    // Update task completion chart
    task_complete_chart.data.datasets[0].data = [m.task_complete_confidence];
    task_complete_chart.update('none'); // don't animate

    // Update colors in activity confidence chart
    // This assumes that the task list and activity classifier are
    // aligned. This will not be the case in the future. 
    var chart = Chart.getChart('activity-conf-chart');
    var idx = task_idx + 1; // This list includes background as id 0

    if( previous_idx > task_idx ) {
      // We are going backwards, remove coolors after current_step
      for(var i=idx+1; i<=previous_idx+1; i++) {
        colors[i] = "rgb(254, 219, 101)"; // yellow
        colors[i+1] = "rgba(0, 104, 199, 1.0)" // blue
      }
    }
    else {
      colors[idx] = "rgba(62, 174, 43, 1.0)"; // green
      colors[idx+1] = "rgb(254, 219, 101)"; // yellow
    }
    colors[chart.data.labels.indexOf("Background")] = "rgba(0, 104, 199, 1.0)"; // blue

    chart.data.datasets[0].backgroundColor = colors;
    chart.update('none'); // don't animate update
  });

});
