// Get data from angel_system
workspace = '/debug';
var ros = new ROSLIB.Ros({
  url : 'ws://localhost:9090'
});

var request = {};
var chart;
var xValues = [];
var yValues = [];
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

// Create a listener for activity detections
var activity_listener = new ROSLIB.Topic({
  ros : ros,
  name : workspace + '/ActivityDetections',
  messageType : 'angel_msgs/ActivityDetection'
});

activity_listener.subscribe(function(m) {
  xValues = m.label_vec;
  yValues = m.conf_vec;

  chart.data.labels = xValues;
  chart.data.datasets[0].data = yValues;

  //chart.options.scales.xAxes[0]

  chart.update('none'); // don't animate
});
