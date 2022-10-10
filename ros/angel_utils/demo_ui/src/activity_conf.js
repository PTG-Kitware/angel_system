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
var chart;
query_task_graph.callService(request, function(result){
  var xValues = result.task_graph.task_steps;
  var yValues = new Array(xValues.length).fill(0);
  var barColors = "rgba(0, 104, 199, 1.0)";

  var ctx = document.getElementById("activity-conf").getContext('2d');

  chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: xValues,
      datasets: [{
        backgroundColor: barColors,
        data: yValues
      }]
    },
    options: {
      scales: {
          xAxes: [{
            ticks: {
                callback: function(t) {
                  return xValues.indexOf(t)
                }
            }
          }],
          yAxes: [{
            ticks: {
              beginAtZero: true,
              max: 1
            }
          }]
      },
      legend: {display: false},
      title: {
        display: false,
      }
    }
  });
});

// Create a listener for activity detections
var activity_listener = new ROSLIB.Topic({
  ros : ros,
  name : workspace + '/ActivityDetections',
  messageType : 'angel_msgs/ActivityDetection'
});

activity_listener.subscribe(function(m) {
  // update chart
  chart.data.labels = m.label_vec;
  chart.data.datasets[0].data = m.conf_vec;
  chart.update('none'); // don't animate
});
