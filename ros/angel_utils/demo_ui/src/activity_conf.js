var xValues = ["Pour 12 ounces of water into liquid measuring cup", "Pour the water from the liquid measuring cup into the electric kettle", "Turn on the kettle", "Turn on kitchen scale", "Place bowl on scale", "Zero scale", "Add coffee beans until scale reads 25 grams", "Pour coffee beans into coffee grinder", "Take the coffee filter and fold it in half to create a semi circle", "Fold the filter in half again to create a quarter circle", "Place the folded filter into the dripper such that the point of the quarter circle rests in the center of the dripper", "Spread the filter open to create a cone inside the dripper", "Place the dripper on top of the mug", "Background"]
var yValues = new Array(xValues.length).fill(0);
var barColors = "rgba(0, 104, 199, 1.0)";

var ctx = document.getElementById("activity-conf").getContext('2d');

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
        xAxes: [{
           ticks: {
              callback: function(t) {
                 return xValues.indexOf(t)
              }
           }
        }],
     },
    legend: {display: false},
    title: {
      display: false,
    }
  }
});

// Get data from angel_system
workspace = '/debug';
var ros = new ROSLIB.Ros({
  url : 'ws://localhost:9090'
});

// Create a listener for /angel/ActivityDetections
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
