var request = {};
var chart;
var xValues = [0];
var yValues = [0];
var barColors = "rgba(0, 104, 199, 1.0)";

Chart.defaults.font.size = 25;
Chart.defaults.font.family = "Roboto","sans-serif";

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
        x: {
          ticks: {
              callback: function(val, index) {
                return val;//return xValues.indexOf(t)
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
    plugins:{
      legend: {display: false}
    },
    title: {
      display: false,
    }
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
