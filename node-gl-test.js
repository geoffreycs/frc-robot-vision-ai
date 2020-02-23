"use strict"; //Because of course we are

//Change your bot's IP or hostname here
//const ip_address = "169.254.106.59"

//Set up the webcam
let NodeCam = require("node-webcam");
let cam_opts = {
    width: 640, //Low resolution so image is small and program is light
    height: 480,
    quality: 70, //Neural network doesn't need that great of a pciture
    saveShots: false, //No need to waste memory by saving
    output: "png", //Format must be JPEG
    device: false, //Use default (only) camera
    callbackReturn: "buffer", //Return as a buffer so we can work with it immediately
    verbose: true //More logging is better
};
let Webcam = NodeCam.create(cam_opts);

//Load Tensorflow.JS
let tf = require('@tensorflow/tfjs');
var model;

//Set up stuff for using GPU processing
//All of this shit is highly experimental
//Stolen from https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-nodegl/src/index.ts
const nodeGles = require('node-gles');
const nodeGl = nodeGles.createWebGLRenderingContext();
tf.env().set('WEBGL_VERSION', 2);
tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);
tf.env().set('WEBGL_DOWNLOAD_FLOAT_ENABLED', true);
tf.env().set('WEBGL_FENCE_API_ENABLED', true); // OpenGL ES 3.0 and higher..
tf.env().set(
    'WEBGL_MAX_TEXTURE_SIZE', nodeGl.getParameter(nodeGl.MAX_TEXTURE_SIZE));
tf.webgl.setWebGLContext(2, nodeGl);

tf.registerBackend('headless-nodegl', () => {
    // TODO(kreeger): Consider moving all GL creation here. However, weak-ref to
    // GL context tends to cause an issue when running unit tests:
    // https://github.com/tensorflow/tfjs/issues/1732
    return new tf.webgl.MathBackendWebGL(new tf.webgl.GPGPUContext(nodeGl));
}, 3 /* priority */ );

//Load HTML5 canvas shims
const {
    createCanvas,
    Image
} = require('canvas');

/*
//Load the NetworkTables client
const ntClient = require('wpilib-nt-client');
const client = new ntClient.Client();
*/

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
var blank_count = 0;
var closest_X = 0.0;
var canvas = createCanvas(cam_opts.width, cam_opts.height);
var ctx = canvas.getContext('2d');

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

function DetectionFilter() { //Filters out all detections not above 0.5 certainty
    function filterDetections(scores) {
        var filtered = new Array();
        scores.forEach((item, index) => {
            item = +item;
            if ((+item) > 0.5) {
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
    model = await tf.loadGraphModel('http://127.0.0.1:8080/model.json'); //Load the re-trained COCO RCNN v2 model
    console.log("Connection state at loop begin is " + String(startState));
    //var keyID = Number(client.getKeyID("/coprocessor/shutdown")); //Grab the NetworkTables ID of the shutdown key
    while (true) { //Loops through this block until the shutdown signal is sent
        try {
            /* if (client.getEntry(keyID) == true) { //Listen for when that shutdown entry switches to true
                running = false; //And disable the loop so the function finally exits
            } */
            data = await new Promise((resolve, reject) => {
                Webcam.capture("image", function (err, data) { //Take the picture
                    if (err) { //Note an error if it occurs
                        reject(err);
                    } else {
                        resolve(data);
                    }
                });
            });
            try { //Place this inside a try{} block just in case it errors
                var loadedImage = await new Promise((resolve, reject) => {
                    var img = new Image();
                    img.onload = function () {
                        resolve(img);
                    };
                    img.onerror = function (err) {
                        reject(error);
                    };
                    console.log("Creating image object");
                    img.src = data;
                });
                ctx.drawImage(loadedImage, 0, 0);
                console.log("Drew image onto canvas");
                var img_tens = tf.browser.fromPixels(canvas);
                console.log("Created image tensor from canvas");
                img_tens = img_tens.expandDims(0); //Expand the Tensor along axis zero (to match image dimensions)
                model_in = {
                    'image_tensor': img_tens
                }; //Pass in the image
                console.log("Preparing to execute model")
                output = await model.executeAsync(model_in, model_out); //And get a response back
                console.log("Model successfully executed");
                detection_boxes_raw = await output[0].data(); //Extract raw box values from the detection_boxes tensor
                detection_boxes_grouped = chunkArray(detection_boxes_raw, 4); //Group the values by detection
                detection_scores = await output[1].data(); //Extract the score values from the detection_scores tensor
                filtered_detections = filterDetections(detection_scores); //Filter out the many, many false positives
                if (filtered_detections.length > 0) { //We can skip this block if no balls were detected
                    blank_count = 0; //Reset the counter of how many iterations we failed to detect any balls
                    merged_boxes = mergeDetected(filtered_detections, detection_boxes_grouped); //Rearrange the data into this handier format
                    cartesian_converted = toCartesian(merged_boxes); //Recenter the X-values so that 0 is the middle of the image
                    closest = findClosest(cartesian_converted); //Find the closest ball based on its radius in the image
                    console.log("Closest ball is at index " + String(closest));
                    closest_X = cartesian_converted[closest].X; //Grab the X-axis value of this ball's centerpoint
                    if (closest_X > 75) { //Turn right if it's on the right of the picture
                        console.log("Turn right");
                        client.Assign("right", "/coprocessor/turn");
                    } else if (closest_X < -75) {
                        console.log("Turn left"); //Ditto but for left
                        client.Assign("left", "/coprocessor/turn");
                    } else {
                        client.Assign("ahead", "/coprocessor/turn"); //Or just go forwards if it's roughly centered
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
                } else {
                    console.log("0 detections");
                    console.log("track = " + String(track));
                    blank_count = blank_count + 1; //Increment this counter
                    console.log("blank_count = " + blank_count);
                    if (blank_count > 1) { //We ignore the first blank because the camera glitches sometimes
                        if (blank_count > 15) { //If there's been this many blanks, we need to change tactics
                            track = track * -1;
                        }
                        if (track > 0) { //Use the last known motion-tracking valuie to determine where to go
                            console.log("Turn right");
                            client.Assign("right", "/coprocessor/turn");
                        } else if (track < 0) {
                            console.log("Turn left");
                            client.Assign("left", "/coprocessor/turn");
                        }
                    }
                }
            } catch (e) {
                console.log(e); //Note the error and continue the loop
                process.exit(-1);
            }
        } catch (rejection) {
            console.log(rejection); //Note any issues with taking the picture
        }
    }
}

var client = {
    Assign: function (value, key) {
        console.log("Key = " + key + "\nValue = " + value);
    }
};

//mainLoop('yes')
tf.setBackend('headless-nodegl').then(() => mainLoop("yes"));

/*
//Bring up the NetworkTables connection
client.start((isConnected, err) => {
    if (err) {
        throw err; //Abort if there is an issue
    } else {
        mainLoop(isConnected).then(
            () => process.exit(0) //Be ready to shutdown the program when we get the signal
        ); //Start actual program loop
    }
}, ip_address); //Specify IP of NetworkTables server (the roboRIO)
*/

//Written by Geoffrey C. Stentiford of Team Voltron, FRC #7064
//https://github.com/geoffreycs/frc-robot-vision-ai