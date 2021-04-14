"use strict"; //Because of course we are

//Define command line options
const optionDefinitions = [
    { name: 'address', alias: 'a', type: String, defaultOption: true, defaultValue: '10.70.64.2' },
    { name: 'width', alias: 'w', type: Number, defaultValue: 640 },
    { name: 'height', alias: 'h', type: Number, defaultValue: 360 }
];

//Parse options
const commandLineArgs = require('command-line-args');
const options = commandLineArgs(optionDefinitions);

//Change your bot's IP or hostname here
const ip_address = options.address;
console.info("IP address is " + ip_address);

//Set up MJpeg capture
const MjpegDecoder = require('mjpeg-decoder');
let cam_opts = {
    width: options.width, //Put in the given image resolution
    height: options.height
};
console.info("Expecting image of size " + String(cam_opts.width) + "x" + String(cam_opts.height));

//Load Tensorflow.JS
let tf = require('@tensorflow/tfjs-node-gpu');
var model;

//Load the NetworkTables client
const ntClient = require('wpilib-nt-client');
const client = new ntClient.Client();

//Pre-reserve some variables
var data = new Buffer(""); //Where the JPEG image will first go
var output = new Array(); //Where the output will be written to eventually
var model_in = {
    "image_tensor": undefined
}; //Input parameters to neural network
const model_out = ['detection_boxes', 'detection_scores']; //Output tensors that we care about
var detection_boxes_raw = new Float32Array(); //For storing flattened outputs from those tensors
var detection_boxes_grouped = new Array();
var detection_scores = new Float32Array();
var filtered_detections = new Array();
var merged_boxes = new Array();
var cartesian_converted = new Array();
var closest = 0 | 0;
var last_X = null;
var track = 0 | 0;
var blank_count = 0 | 0;
var hit_count = 0 | 0;
var closest_X = 0.0;
var detected_once = false;

//Compile some asm.js modules for even more speed
function CenterPoint(stdlib, foreign, buffer) { //Finds the average of two values
    'use asm';

    function findCenter(x, y) {
        x = +x;
        y = +y;
        var c = 0.0;
        c = +(+(+x + +y) / 2.0);
        return +c;
    }
    return findCenter;
}
let findCenter = new CenterPoint({}, null, new ArrayBuffer(0x10000));

function Radius(stdlib, foreign, buffer) { //Finds the radius of the ball
    'use asm';

    function findRadius(Xcent, Ycent, Xmax, Ymax) {
        //console.log("Xcent = " + Xcent);
        //console.log("Ycent = " + Ycent);
        //console.log("Xmax = " + Xmax);
        //console.log("Ymax = " + Ymax);
        Xcent = +Xcent;
        Ycent = +Ycent;
        Xmax = +Xmax;
        Ymax = +Ymax;
        var Xdiff = 0.0;
        var Ydiff = 0.0;
        Xdiff = +((+Xmax) - (+Xcent));
        //console.log("Xdiff = " + Xdiff);
        Ydiff = +((+Ymax) - (+Ycent));
        //console.log("Ydiff = " + Ydiff)
        var final = 0.0;
        if ((+Xdiff) > (+Ydiff)) {
            //console.log("Xdiff > Ydiff");
            final = (+Xdiff);
        } else {
            //console.log("Ydiff > Xdiff");
            final = (+Ydiff);
        }
        //console.log("final = " + final);
        return +final;
    }
    return findRadius;
}
let findRadius = new Radius({}, null, new ArrayBuffer(0x10000));

//Force pre-interpret a few handy functions for a tiny bit more speed
function ArrayChunk() { //Breaks arrays up into sub-arrays of equal size
    //Stolen from
    // ourcodeworld.com/articles/read/278/how-to-split-an-array-into-chunks-of-the-same-size-easily-in-javascript
    function chunkArray(myArray, chunk_size) {
        var index = 0;
        var arrayLength = myArray.length;
        var tempArray = [];
        for (index = 0; index < arrayLength; index += chunk_size) {
            var myChunk = myArray.slice(index, index + chunk_size);
            tempArray.push(myChunk);
        }

        return tempArray;
    }
    return chunkArray;
}
let chunkArray = new ArrayChunk();

function DetectionFilter() { //Filters out all detections not above 0.6 certainty
    function filterDetections(scores) {
        var filtered = new Array();
        scores.forEach((item, index) => {
            item = +item;
            if ((+item) > 0.6) {
                console.log("Adding index " + String(index))
                filtered.push(index);
            }
        });
        return filtered;
    }
    return filterDetections;
}
let filterDetections = new DetectionFilter();

function DetectedMerger(cam_opts, foreign) { //Takes filtered detections and reformats that data
    var findCenter = foreign.findCenter; //Link the imported (asm.js) functions
    var findRadius = foreign.findRadius;
    var temp_center_x = 0.0;
    var temp_center_y = 0.0;
    var img_height = 0.0;
    var img_width = 0.0;
    img_height = +cam_opts.height;
    img_width = +cam_opts.width;

    function mergeDetected(filtered_detections, detection_boxes_grouped) {
        var merged_boxes = [];
        var temp_box = {
            ymin: 0.0,
            xmin: 0.0,
            ymax: 0.0,
            xmax: 0.0
        };
        filtered_detections.forEach((item) => { //Loop through and scale up box to match image size
            temp_box.ymin = +detection_boxes_grouped[item][0] * +img_height;
            temp_box.xmin = +detection_boxes_grouped[item][1] * +img_width;
            temp_box.ymax = +detection_boxes_grouped[item][2] * +img_height;
            temp_box.xmax = +detection_boxes_grouped[item][3] * +img_width;
            temp_center_x = +findCenter(temp_box.xmin, temp_box.xmax);
            temp_center_y = +findCenter(temp_box.ymin, temp_box.ymax);
            merged_boxes.push({
                Xcenter: +temp_center_x, //Then find their centerpoints
                Ycenter: +temp_center_y,
                radius: +findRadius(+temp_center_x, +temp_center_y, +temp_box.xmax, +temp_box.ymax) //And find their radii
            });
        });
        return merged_boxes;
    }
    return mergeDetected;
}
let mergeDetected = new DetectedMerger({
    height: cam_opts.height,
    width: cam_opts.width
}, {
    findCenter: findCenter, //Link these two functions internally for performance
    findRadius: findRadius
});

function CartesianConvert(cam_opts) { //Recenters the X-values so that 0 is the image center
    var img_width = 0.0;
    img_width = +cam_opts.width;
    var shift_x = 0.0;
    shift_x = +((+img_width) / 2.0); //Calculate how much to shift by dividing the image width in half

    function toCartesian(merged_boxes) {
        var converted = [];
        merged_boxes.forEach((item) => { //Loop through each box object in the array
            converted.push({
                X: (+item.Xcenter) - (+shift_x), //Perform the X-axis shift
                r: +item.radius //And tack on the radius for convenience
            });
        });
        return converted;
    }
    return toCartesian;
}
let toCartesian = new CartesianConvert({
    height: cam_opts.height,
    width: cam_opts.width
});

function ClosestBall(stdlib) { //Figure out which ball is closest from their radii
    var max = stdlib.Math.max; //Import the linked Math function

    function findClosest(cartesian_converted) {
        var radii = [];
        var answer = 0 | 0;
        var max_r = 0.0;
        cartesian_converted.forEach((item) => { //Loop through and copy all the radii to a new array
            radii.push(item.r)
        });
        max_r = max(...radii); //Get the largest value in this new array
        cartesian_converted.forEach((item, index) => { //Loop through the first array again
            if (item.r == max_r) { //and compare the radius value of each element to the known largest one
                answer = index; //Record the positional index of the element with the matching radius
            }
        });
        return answer | 0;
    }
    return findClosest;
}
let findClosest = new ClosestBall({
    Math: { //Link the Math.max function internally for performance
        max: Math.max
    }
});

//Meat of the script
async function mainLoop(startState) {
    var running = true;
    model = await tf.loadGraphModel('file://model/model.json'); //Load the re-trained COCO RCNN v2 model
    console.log("Connection state at loop begin is " + String(startState));
    var MjpegStream = client.getEntry(client.getKeyID("/CameraPublisher/USB Camera 0/streams")).val[1].substring(5); //Get the camera stream's URL
    var decoder;
    //var keyID = Number(client.getKeyID("/coprocessor/shutdown")); //Grab the NetworkTables ID of the shutdown key
    while (true) { //Loops through this block
        tf.engine().startScope(); //Use a scope so memory is freed back up after every cycle
        console.log("\n");
        try {
            decoder = new MjpegDecoder(MjpegStream, { maxFrames: 1 });
            data = await decoder.takeSnapshot(); //Take a frame from the camera stream
            try { //Place this inside a try{} block just in case it errors
                var converted = new Uint8Array(data); //Turn the Node Buffer into an Uint8Array
                var img_tens = tf.node.decodeJpeg(converted); //Turn the Uint8Array into a 3D Tensor
                img_tens = img_tens.expandDims(0); //Expand the Tensor along axis zero (to match image dimensions)
                model_in = {
                    'image_tensor': img_tens
                }; //Pass in the image
                output = await model.executeAsync(model_in, model_out); //And get a response back
                detection_boxes_raw = await output[0].data(); //Extract raw box values from the detection_boxes tensor
                detection_boxes_grouped = chunkArray(detection_boxes_raw, 4); //Group the values by detection
                detection_scores = await output[1].data(); //Extract the score values from the detection_scores tensor
                filtered_detections = filterDetections(detection_scores); //Filter out the many, many false positives
                if (filtered_detections.length > 0) { //We can skip this block if no balls were detected
                    blank_count = 0 | 0; //Reset the counter of how many iterations we failed to detect any balls
                    hit_count = hit_count + 1; //Increment the counter how many interactions we did find a ball
                    detected_once = true; //Trigger the fuse to show that we have detected at least once so far
                    console.log("hit_count = " + String(hit_count));
                    merged_boxes = mergeDetected(filtered_detections, detection_boxes_grouped); //Rearrange the data into this handier format
                    cartesian_converted = toCartesian(merged_boxes); //Recenter the X-values so that 0 is the middle of the image
                    closest = findClosest(cartesian_converted); //Find the closest ball based on its radius in the image
                    console.log("Closest ball is at index " + String(closest));
                    closest_X = cartesian_converted[closest].X; //Grab the X-axis value of this ball's centerpoint
                    console.log("Certainty of closest detection is " + String(detection_scores[closest].toFixed(4) * 100) + "%");
                    client.Assign(detection_scores[closest].toFixed(4) * 100, "/coprocessor/certainty"); //Push the certainty to NetworkTables
                    console.log("Radius of closest detection is " + String(cartesian_converted[closest].r));
                    client.Assign(cartesian_converted[closest].r, "/coprocessor/radius"); //Push the ball's relative size to NetworkTables
                    client.Assign(false, "/coprocessor/guessing"); //Indicate to the bot that the ball is being actively tracked
                    if (closest_X > 25) { //Turn right if it's on the right of the picture
                        console.log("Turn right");
                        client.Assign(1.0, "/coprocessor/turn");
                    } else if (closest_X < -25) {
                        console.log("Turn left"); //Ditto but for left
                        client.Assign(-1.0, "/coprocessor/turn");
                    } else {
                        client.Assign(0.0, "/coprocessor/turn"); //Or just go forwards if it's roughly centered
                        console.log("Ahead");
                    }
                    console.log("last_X = " + String(last_X));
                    console.log("closest_X = " + String(closest_X));
                    if (last_X != null) {
                        //if (Math.abs(last_X - closest_X) < 80) {
                        if (last_X > closest_X) { //If it's left of its last known position, it's moving left
                            track = -1 | 0;
                        } else {
                            track = 1 | 0; //Again, likewise for the right side
                        }
                        console.log("track = " + String(track));
                        //}
                    }
                    last_X = closest_X; //Update this now old value
                } else { //Or if we can't find a ball...
                    console.log("0 detections");
                    if (detected_once == false && blank_count == 0) { //Check if we've gotten even a single detection so far
                        if ((Math.random() - 0.5) > 0) { //Then give a random motion tracking value so the bot doesn't just sit there
                            track = 1;
                        } else {
                            track = -1;
                        }
                    }
                    console.log("track = " + String(track));
                    blank_count = blank_count + 1; //Increment this counter
                    hit_count = 0 | 0;
                    console.log("blank_count = " + String(blank_count));
                    client.Assign(0, "/coprocessor/certainty"); //Set the certainty key to zero because we lost the ball
                    client.Assign(true, "/coprocessor/guessing"); //Let the bot know that the AI ain't seein' sHIT
                    client.Assign(0.0, "/coprocessor/radius");
                    if (blank_count == 40) { //If there's been this many blanks, we need to reverse direction
                        track = track * -1;
                        console.log("Reversing due to 40 consecutive blanks");
                    }
                    if (track > 0) { //Use the last known motion-tracking value to determine where to go
                        console.log("Turn right");
                        client.Assign(1.0, "/coprocessor/turn");
                    } else if (track < 0) {
                        console.log("Turn left");
                        client.Assign(-1.0, "/coprocessor/turn");
                    }
                }
                client.Assign(blank_count, "/coprocessor/blanks"); //Push these counters to the NetworkTable
                client.Assign(hit_count, "/coprocessor/hits");
            } catch (e) {
                console.error(e); //Note the error and continue the loop
                //throw e;
            }
        } catch (rejection) {
            //console.log(rejection); //Note any issues with taking the picture
            throw rejection;
        }
        tf.engine().endScope();
    }
}

//Bring up the NetworkTables connection
client.start((isConnected, err) => {
    if (err) {
        throw err; //Abort if there is an issue
    } else {
        //MjpegStream = "http://10.70.64.2:1181/?action=stream"
        mainLoop(isConnected).then(
            () => process.exit(0) //Be ready to shutdown the program when we get the signal
        ); //Start actual program loop
    }
}, ip_address); //Specify IP of NetworkTables server (the roboRIO)


//Written by Geoffrey C. Stentiford of Team Voltron, FRC #7064
//https://github.com/geoffreycs/frc-robot-vision-