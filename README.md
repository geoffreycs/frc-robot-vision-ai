# frc-robot-vision-ai
This is a Node.JS script that tracks the location of yellow balls for FRC Infinite Recharge. It's meant to be run on some sort 
of coprocessor device, such as a Jetson or a Raspberry Pi (or a full-fledged x86_64 PC with a desktop CPU like that one team did).   
## Requirements   
* Node.js, of course
* `npm` for installing the dependencies
* `fswebcam` for taking pictures via `node-webcam`
* A CPU capable of running TensorFlow.js models with some level of performance
  * If you're running this (for some reason) on a Linux system with a CUDA-capable GPU, you can switch out the `@tensorflow/tfjs-node`
  with `@tensorflow/tfjs-node-gpu` in `package.json`. In that case, you'll also need to install/configure CUDA support
* A writable (but not necessarily non-volatile) filesystem for the program to run on
  * An image is continuously saved to `image.jpg` in the root folder. I recommend using the `overlay` filesystem, with the actual 
  root directory being read-only and any changes being stored in RAM.
* A roboRIO configured to use this coprocessor script   
## Usage
1. Clone this repo: `git clone https://github.com/geoffreycs/frc-robot-vision-ai`
2. (optional) Switch `@tensorflow/tfjs-node` to `@tensorflow/tfjs-node-gpu` in `package.json`
3. Do `npm install`
4. Edit `main.js` and set the right robot hostname or IP address in the bottom of the file
5. Set up some method to auto-run `node main.js` when needed
