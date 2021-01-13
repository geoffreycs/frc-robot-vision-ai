# frc-robot-vision-ai
This is a Node.JS script that tracks the location of yellow balls for FRC Infinite Recharge. It's meant to be run on some sort 
of coprocessor device, such as a Jetson or a Raspberry Pi (or a full-fledged x86_64 PC with a desktop CPU/GPU like that one team did).   
## Requirements   
* Node.js, of course
* `npm` for installing the dependencies
* `fswebcam` for taking pictures via `node-webcam`
* `mjpeg-decoder` for capturing images from the robot's MJPEG camera stream (used in `computer-side.js`)
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
3. Do `npm install` or `yarn install`
4. Edit `main.js` and set the right robot hostname or IP address in the top of the file
5. Set up some method to auto-run `node main.js` when needed
## Secondary files
### `computer-side.js`
`computer-side.js` is functions much like `main.js`, except it is designed to run on the driver station computer instead. It uses the CPU version of NodeJS by default, although it's a simple change to make it run on the GPU. Instead of using a local webcam, it captures frames from the robot's MJEPG camera stream.
### Backend test files
The numerous other `.js` files are mostly identical to `main.js`, but they:
* Do not load the NetworkTables client
* Replace the NetworkTables key-value assign function with a `console.log();`
* Use different Tensorflow.js backends
* Require `devDependencies` to be installed
* Generally do not work besides `node-gpu.js` (which uses the native CUDA backend and so will generally not be usable on a robot-mounted coprocessor) and `cpu-vanilla-test.js` (which uses the pure JavaScript backend and so is more compatible but absurdly slow
* (Excludes `node-gpu.js`) Require an HTTP server running at `127.0.0.1:8080` and serving the `model` directory aa the webserver root (a modified `nweb` is provided as both source and x86_64 Linux ELF binary for this purpose)
* (Excludes `node-gpu.js`) Take the image as a PNG as part of a workaround to build an image tensor without `tf.node.decodeJpeg()`, which isn't present in `@tensorflow/tfjs`.
### `nt_client_test.js`
Simple script to toggle a value on NetworkTables at `/testtable/testkey` to test the NetworkTables setup.
### nweb
Needed to run all backend tests besides `node-gpu.js`. The compiled binary is an x86_64 ELF for Linux. It is dynamically linked, but will run on most common setups. You can recompile if needed with `gcc -O -DLINUX nweb23.c -o nweb`. Run using `.nweb 8080 ./model`. This version is slightly modified to also serve the model's `.bin` files with the `application/octet-stream` MIME type.
