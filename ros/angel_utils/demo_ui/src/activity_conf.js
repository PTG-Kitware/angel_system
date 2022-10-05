var xValues = ["Take the coffee filter and fold it in half to create a semi circle", "Fold the filter in half again to create a quarter circle", "Place the folded filter into the dripper such that the point of the quarter circle rests in the center of the dripper", "Spread the filter open to create a cone inside the dripper", "Place the dripper on top of the mug"]
var yValues = [0.6894994132599095, 0.9350718503119424, 0.0715317816666357, 0.455273289709737, 0.1832097445440013, 0.9988147020339966];
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
