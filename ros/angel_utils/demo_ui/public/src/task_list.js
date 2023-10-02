var task_step_conf_ctx = document.getElementById("task-step-conf-chart").getContext('2d');
var task_complete_ctx = document.getElementById("task-complete-chart").getContext('2d');

var task_step_conf_chart = new Chart(task_step_conf_ctx, {
    type: "bar",
    data: {
        labels: ["task step confidence"],
        datasets: [{
            backgroundColor: "rgba(0, 104, 199, 1.0)",
            data: [0]
        }]
    },
    options: {
        scales: {
            x: {
                ticks: {
                    callback: function(val, index) {
                        return val;
                    }
                }
            },
            y: {
                min: 0,
                max: 1,
                ticks: {
                    stepSize: 0.1
                }
            }
        },
        plugins: {
          title: {
            display: true,
            text: 'Step',
            color: barColors,
            font:{
              size: 15
            },
            position: 'top'
          }
        },
        maintainAspectRatio: false
    }
});

var task_complete_chart = new Chart(task_complete_ctx, {
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

$.get("/topics")
.done(function( topics ){
  var colors;

  // subscribe to QueryTaskGraph
  var query_task_graph = new ROSLIB.Service({
    ros: ros,
    name: topics.namespace + '/' + topics.query_task_graph_topic,
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
    name : topics.namespace + '/' + topics.task_updates_topic,
    messageType : 'angel_msgs/msg/TaskUpdate' // find: $ ros2 topic type <>
  });

  task_update.subscribe(function(m) {
    const completed_steps = m.completed_steps;
    completed_steps.forEach(function(completed, index){
      let el = document.getElementById(task_list[index]);
      // Update checkmarks in task list
      if(completed) {
        el.querySelector('.checkmark').className = 'checkmark_visible checkmark';
        colors[index+1] = "rgba(62, 174, 43, 1.0)"; // green
      }
      else {
        el.querySelector('.checkmark').className = 'checkmark_hidden checkmark';
        colors[index+1] = "rgba(0, 104, 199, 1.0)"; // blue
      }
    });

    // Update task step conf chart with done-color association.
    task_step_conf_chart.data.labels = [...Array(m.hmm_step_confidence.length).keys()];
    task_step_conf_chart.data.datasets[0].data = m.hmm_step_confidence;
    task_step_conf_chart.data.datasets[0].backgroundColor = colors;
    task_step_conf_chart.update('none'); // don't animate

    // Display highest conf in title
    max_val = Math.max(...task_step_conf_chart.data.datasets[0].data);
    max_idx = task_step_conf_chart.data.datasets[0].data.indexOf(max_val);

    new_title = task_step_conf_chart.data.labels[max_idx];
    task_complete_chart.options.plugins.title.text = new_title;

    // Update task completion chart
    task_complete_chart.data.datasets[0].data = [m.task_complete_confidence];
    task_complete_chart.update('none'); // don't animate
  });

});
