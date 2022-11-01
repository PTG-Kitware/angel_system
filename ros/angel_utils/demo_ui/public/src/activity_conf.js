var request = {};
var xValues = [0];
var yValues = [0];
var barColors = "rgba(0, 104, 199, 1.0)";

var ctx = document.getElementById("activity-conf-chart").getContext('2d');

var chart = new Chart(ctx, {
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
    title: {
      display: false,
    },
    maintainAspectRatio: false
  }
});

$.get( "/ns")
.done(function( data ){
  // Create a listener for activity detections
  var activity_listener = new ROSLIB.Topic({
    ros : ros,
    name : data.namespace + '/ActivityDetections',
    messageType : 'angel_msgs/ActivityDetection'
  });

  activity_listener.subscribe(function(m) {
    xValues = m.label_vec;
    yValues = m.conf_vec;

    chart.data.labels = xValues;
    chart.data.datasets[0].data = yValues;

    chart.update('none'); // don't animate
  });
});
