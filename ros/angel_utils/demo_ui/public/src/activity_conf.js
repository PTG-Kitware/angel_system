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
    plugins: {
      title: {
        display: true,
        text: 'Activity',
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

$.get( "/topics")
.done(function( topics ){
  // Create a listener for activity detections
  var activity_listener = new ROSLIB.Topic({
    ros : ros,
    name : topics.namespace + '/' + topics.activity_detections_topic,
    messageType : 'angel_msgs/ActivityDetection'
  });

  activity_listener.subscribe(function(m) {
    xValues = m.label_vec;
    yValues = m.conf_vec;

    chart.data.labels = xValues;
    chart.data.datasets[0].data = yValues;

    // Display highest conf in title
    max_val = Math.max(...yValues);
    max_idx = yValues.indexOf(max_val);

    new_title = xValues[max_idx];
    chart.options.plugins.title.text = new_title;

    chart.update('none'); // don't animate
  });
});
