var recipe_colors = ["blue", "green", "yellow", "red", "orange"]
const zip = (a, b) => a.map((k, i) => [k, b[i]]);

var ctx = document.getElementById("task-step-chart").getContext('2d');
var task_step_chart = new Chart(ctx, {
    type: "line",
    data: {
        labels: [],
        datasets: []
    },
    options: {
        plugins: {
            legend: {
                display: true,
                labels: {
                    fontSize: 10,
                }
            },
            title: {
                display: false
            }
        },
        maintainAspectRatio: false,
        scales: {
            x: {
                ticks: {
                    display: false
                },
                title: {
                    display: true,
                    text: "Time"
                }
            },
            y: {
                stacked: false,
                title: {
                    display: true,
                    text: "Step Number"
                },
                min: 0,
                max: 25,
                ticks: {
                    stepSize: 1
                }
            }
        }
    }
});

$.get("/topics")
.done(function( topics ){
  // subscribe to QueryTaskGraph
  var query_task_graph = new ROSLIB.Service({
    ros: ros,
    name: topics.namespace + '/' + topics.query_task_graph_topic,
    serviceType: 'angel_msgs/srv/QueryTaskGraph'
  });

  var request = {};
  var task_title_to_steps = {};
  var step_list;
  query_task_graph.callService(request, function(result){
    
    var task_titles = result.task_titles;
    var task_graphs = result.task_graphs;

    // TODO: Check that we have enough fields
    // Or maybe dynamically insert fields for tasks?
    
    var tasks = zip(task_titles, task_graphs);
    tasks.forEach(function(task, index){
        var task_color = recipe_colors[index];

        // Load title
        var task_title = task[0];
        task_title_to_steps[task_title] = {}
        var title_container_block = document.getElementById(task_color + '-task_title');
        title_container_block.innerHTML = task_title;

        // Add recipe to graph
        task_step_chart.data.datasets.push({
            label: task_title,
            data: [],
            borderColor: task_color,
            backgroundColor: task_color,
            yAxisID: 'y'
        });
        task_title_to_steps[task_title]["chart_id"] = task_step_chart.data.datasets.length - 1

        // Load task graph
        var task_graph = task[1];

        step_list = task_graph.task_steps;
        task_title_to_steps[task_title]['steps'] = step_list;
        var task_levels = task_graph.task_levels;
        var container_block = document.getElementById(
            task_color + '-task-list'
        );
        container_block.className += " " + task_title;

        // Load steps
        var step_list = task_graph.task_steps;
        step_list.forEach(function(step, index){
            // TODO: support different step levels
            var step_level = task_levels[index];
      
            var step_line = document.createElement('div');
            step_line.className = "task-line";
      
            var checkbox = document.createElement('span');
            checkbox.className = "checkbox";
            checkbox.id = step;
            step_line.appendChild(checkbox);
      
            var text = document.createElement('span');
            text.className = "text body-text step";
            text.innerHTML = index+1 + '. ' + step;
            step_line.appendChild(text);
      
            container_block.appendChild(step_line);
          });
    });
  });

  // Create a listener for task completion updates
  var task_update = new ROSLIB.Topic({
    ros : ros,
    name : topics.namespace + '/' + topics.task_updates_topic,
    messageType : 'angel_msgs/msg/TaskUpdate' // find: $ ros2 topic type <>
  });

  task_update.subscribe(function(m) {
    var task_name = m.task_name;
    var step_list = task_title_to_steps[task_name]["steps"];
    var chart_id = task_title_to_steps[task_name]["chart_id"];

    const completed_steps = m.completed_steps;
    completed_steps.forEach(function(completed, index){
      let box = document.getElementById(step_list[index]);

      // Update boxes in task list
      if(completed) {
        box.style.backgroundColor = "green";
      }
      else {
        box.style.backgroundColor = "white";
      }
    });

    // Update line chart
    var current_step_id = m.current_step_id + 1; // list doesn't include background
    var ts = m.header.stamp.sec;

    if(current_step_id != null){
        task_step_chart.data.datasets.forEach(
            function(dataset, index){
                if(index == chart_id){
                    task_step_chart.data.datasets[chart_id].data.push(current_step_id);
                    console.log(task_step_chart.data.datasets[index].data);
                }
                else{
                    var last_val = task_step_chart.data.datasets[index].data.slice(-1).pop();
                    if(last_val == null){
                        last_val = 0;
                    }
                    console.log(last_val);
                    
                    task_step_chart.data.datasets[index].data.push(last_val);
                    console.log(task_step_chart.data.datasets[index].data);
                }
            }
          );

        task_step_chart.data.labels.push(ts);
    
        task_step_chart.update('none'); // don't animate
    }
    

    
  });
});