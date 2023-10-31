const express = require('express'); //Import the express dependency
var path = require('path');
const commandLineArgs = require('command-line-args');
const app = express();              //Instantiate an express app, the main work horse of this server
const port = 5001;                  //Save the port number where your server will be listening

// Add stylesheets
app.use(express.static(path.join(__dirname, 'public')));

// Get topic names from command line args
const argOpts = [
  { name: 'namespace', alias: 'n', type: String },
  { name: 'image_topic', alias: 'i', type: String },
  { name: 'query_task_graph_topic', alias: 'q', type: String },
  { name: 'task_updates_topic', alias: 't', type: String },
  { name: 'activity_detections_topic', alias: 'a', type: String },
  { name: 'task_errors_topic', alias: 'e', type: String }
];

const args = commandLineArgs(argOpts);
console.log('args:');
console.log(args);

//var ns_json = Object.assign(namespace_json, args);
// Pass the topic name parameters
app.get('/topics', (req, res) => {
    res.json(args);
});

//Idiomatic expression in express to route and respond to a client request
app.get('/', (req, res) => {        //get requests to the root ("/") will route here
    res.sendFile(__dirname +'/main.html');      //server responds by sending the index.html file to the client's browser
});

app.listen(port, () => {            //server starts listening for any attempts from a client to connect at port: {port}
    console.log(`Now listening on port ${port}: http://localhost:${port}`);
});
