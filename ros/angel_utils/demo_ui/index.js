const express = require('express'); //Import the express dependency
var path = require('path');
const app = express();              //Instantiate an express app, the main work horse of this server
const port = 5000;                  //Save the port number where your server will be listening

// Add stylesheets
app.use(express.static(path.join(__dirname, 'public')));

var namespace = process.env.ROS_NAMESPACE;
console.log('ROS Namespace: ' + namespace)
app.get('/ns', (req, res) => {
    res.json({ namespace: namespace });
});

//Idiomatic expression in express to route and respond to a client request
app.get('/', (req, res) => {        //get requests to the root ("/") will route here
    res.sendFile(__dirname +'/main.html');      //server responds by sending the index.html file to the client's browser
                                                    //the .sendFile method needs the absolute path to the file, see: https://expressjs.com/en/4x/api.html#res.sendFile 
});

app.listen(port, () => {            //server starts listening for any attempts from a client to connect at port: {port}
    console.log(`Now listening on port ${port}`); 
});

