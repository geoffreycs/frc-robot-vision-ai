# frc-robot-vision-ai
This is a Node.JS script that tracks the location of yellow balls for FRC Infinite Recharge. It's meant to be run on some sort 
of coprocessor device, such as a Jetson or a Raspberry Pi (or a full-fledged x86_64 PC with a desktop CPU/GPU like that one team did).   
## Requirements   
* Node.js, of course
* `npm` for installing the dependencies
* A CPU or Nvidia GPU capable of running TensorFlow.js models with some level of performance
* An FRC robot with a webam configured to be of sufficient resolution aimed in front or behind the robot and in view of the floor
* Code on the robot written to make use of this program's output
## Webcam
On our robot, we set the camera to 640x360 for the FRC 2021 season. It only got seven frames per second, however, the neural network is the bottleneck, so that wasn't an issue. If you have a fast enough computer, then the low framerate may become an issue. However, if the resolution is too low, then the network fails to recognize any balls. You must specify the same camera resolution in both the the robot code and the command line paramters for this program.
## Usage
1. Clone this repo: `git clone https://github.com/geoffreycs/frc-robot-vision-ai`
2. (optional) Switch `@tensorflow/tfjs-node` to `@tensorflow/tfjs-node-gpu` in `package.json`
3. Do `npm install` or `yarn install`
4. Turn on the robot and connect the device running this program to the robot
5. Start the program with `node main.js [IP address of robot] -w [image_width] -h [image_height]`
## Syntax
```
npm start -- [--address] IP_address [--width webcam_width] [--height webcam_height]
```
Example: `npm start -- 10.70.64.2 -w 640 -h 360`.

If no parameters are specified, then the program defaults to the values in the example above. No, those probably won't be useful for you (because you're likely not Team #7064), but they were quite useful for us.
## API
The program reads and writes to a fixed set of NetworkTable keys.
It reads the URL of the camera stream from `/CameraPublisher/USB Camera 0/streams`, so the camera to be used must be published via the normal WPILib camera server. Every loop iteration, the following keys are written to:
* `/coprocessor/turn` - Integer indicating which direction to turn, with -1 being left, 1 being right, and 0 being straight ahead
* `/coprocessor/certainty` - Double holding the percentage certainty of the detection, or zero if the ball is not within the field of view
* `/coprocessor/radius` - Double with the apparent radius (in pixels) of the closest ball, or zero if the ball is not within the field of view
* `/coprocessor/guessing` - Boolean indicating whether the program is relying on the previously tracked motion of the ball (i.e if the ball is not in view)
* `/coprocessor/hits` - Integer count of consecutive loop iterations that the program has detected a ball
* `/coproessor/blanks` - Integer count of consecutive loop iterations that the program has failed to detect a ball
## Program behavior
On startup, the program first connects to the NetworkTables server at the specified IP address. Next, it reads and then stores the camera stream URL, and also loads the TensorFlow model. It then begins looping though the following process:
1. Capture the current frame from the stream URL
2. Convert the image to a tensor
3. Pass the tensor to TensorFlow and run the model on it
4. Parse the resulting output and filter out all detections below 60% certainty (and if none are found, skip to step 10)
6. Calculate the radii of each filtered detection
7. Store the index of the detection with the largest radii
8. Determine whether the ball is to the left or right of a zone around the center of the image
9. Compare the X-position of the detection with the X-position of the previous iteration's detection in order to determine whether the ball is moving left or right relative to the camera, and store the resulting motion tracking value (and skip to step 12) 
10. If there has been no previous loop iteration yet, randomly assign left or right to the motion tracking value
11. If no balls were found, check the motion tracking value to determine whether to turn left or right
12. Push all the values to NetworkTables
## `nt_client_test.js`
Simple script to toggle a value on NetworkTables at `/testtable/testkey` to test the NetworkTables setup. Run with `node nt_client_test.js {IP_address}`.
