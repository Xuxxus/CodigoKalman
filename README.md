# OpenSense Data Processing

## Overview

This Python code, intended for use in Google Colab, is part of a larger project for sensor data processing. The main objective is to take a text file containing Inertial Measurement Unit (IMU) data, apply Kalman filtering to convert the data into quaternions and Euler angles, and generate the necessary OpenSim files to execute movements in the software.

### Lab-Version Tag

The `Lab-Version` tag represents a version of the code used in the project, demonstrating effective data processing.

## Getting Started

### Prerequisites

- Google Colab account
- Text file containing IMU data
- Python 3.11

### Usage

1. Upload the text file containing IMU data to your Google Colab environment.

2. Open the Jupyter notebook (`main_processing.ipynb`) in Google Colab.

3. Execute the notebook cells sequentially.

4. Monitor the output for any errors or warnings during the processing.

## Kalman Filtering

The code incorporates Kalman filtering to enhance the accuracy of IMU data by estimating quaternions and Euler angles.

## Contributing

If you want to contribute to this data processing code, follow these steps:

1. Fork the repository.

2. Create a new branch for your feature:

   ```bash
   git checkout -b feature-name

3. Make your changes and commit them:
   
   ```bash
   git commit -m 'Add your feature'

4. Push to the branch:

   ```bash
   git push origin feature-name

5. Create a pull request. Please, be clear about your changes.

### Additional Repositories
[Embedded System Repository](https://github.com/Xuxxus/MovementSensoring): Repository for the Arduino code.

[Arduino Calibration Repository](https://github.com/Xuxxus/Calibracao-Accel): Repository for the IMU calibration using Arduino.

[PCB Repository](https://github.com/Xuxxus/ConnectionBoard): Repository for the PCB used in this project.

## Acknowledgments
Professors Wellington Pinheiro, MSc. and Maria Claudia Castro, PhD. from FEI University Center for their absolute great job on guiding our project.


