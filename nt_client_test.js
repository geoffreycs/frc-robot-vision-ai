//Load the NetworkTables client
const ntClient = require('wpilib-nt-client');
const client = new ntClient.Client();

//Parse IP address from command line arguments
const commandLineArgs = require('command-line-args');
const options = commandLineArgs([{ name: 'address', alias: 'a', type: String, defaultOption: true, defaultValue: '10.70.64.2' }]);

//Bring up the NetworkTables connection
client.start((isConnected, err) => {
    if (err) {
        throw err; //Abort if there is an issue
    } else {
        client.Assign("test", "/testtable/testkey");
        console.log("Success");
        client.Assign("again", "/testtable/testkey");
        setTimeout(() => {
            client.destroy();
            process.exit(0);
        }, 3000);
    }
}, options.address);