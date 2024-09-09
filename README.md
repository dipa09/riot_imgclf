This project implement an image classifier using the [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) dataset,
which contains more that 8000 images of 8 different weeds' species.
The goal is to evaluate the performance of some TinyML libraries on RIOT OS.
The libraries that have been selected are: [emlearn](https://github.com/emlearn/emlearn),
[micromlgen](https://github.com/eloquentarduino/micromlgen) and
[m2cgen](https://github.com/BayesWitnesses/m2cgen) and they have been tested on
Arduino Mega and ESP32-CAM boards.


## Dependencies
1. Download the [dataset](https://github.com/AlexOlsen/DeepWeeds).
2. Install the python dependencies listed in `requirements.txt`.
3. Install the toolchain packages required by RIOT for each board.
   Consult [RIOT's documentation](https://api.riot-os.org/getting-started.html) for how to do it.


## Usage
```
# Build tools needed for the traning phase.
$ make driver liblbp.so

# Create a custom dataset.
$ python3 ./prepare.py

# Extract all features.
$ python3 ./extract.py

# Train the model.
$ python3 ./train.py

# Export the model to C.
$ python3 ./generate.py [-l library]

# Extract features for a test image
$ python3 ./extract.py -i <image-path>
$ ./driver features <float|double> features.h

# Build for Linux (optional)
$ make main [LIB=LIBRARY]

# Copy the project to the RIOT application (edit 'Makefile' or set RIOT_DIR)
$ make sync

# From the RIOT application directory
$ make [BOARD=...]

# Connect the board and run
$ make flash [BOARD=...]
```

The preapration, extraction and training steps can be customized, that's why the workflow is
so fragmented. Please use `--help` for more information on how to use the various tools.
Reading the [RIOT documentation](https://api.riot-os.org/creating-an-application.html) for creating
new apps is also recommended.


## Tested version
This project has been tested using the following packages' versions. If you have issues try to
install the specific package version.

| Package       | Version   |
|---------------|-----------|
| emlearn       | 0.20.4    |
| h5py          | 3.11.0    |
| joblib        | 1.4.2     |
| m2cgen        | 0.10.0    |
| mahotas       | 1.4.15    |
| matplotlib    | 3.9.0     |
| micromlgen    | 1.1.28    |
| numpy         | 1.26.4    |
| opencv-python | 4.10.0.84 |
| scikit-learn  | 1.5.0     |


## License
