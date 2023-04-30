# genistat_TheTalkingTrees - Solar Energy Potential Estimation in Germany

This project aims to estimate the solar energy potential of buildings in Germany by analyzing satellite images and OpenStreetMap data to create a dataset of roof shapes. We use a Convolutional Neural Network (CNN) model to detect roof shapes and Computer Vision (CV) techniques to determine roof azimuth angles. By combining this information with sun radiation data and tilt angles, we estimate the electrical potential and efficiency of solar panels on these roofs.

## Project Overview

1. **Data Collection and Preprocessing**: We collect satellite images and OpenStreetMap data to create images of roof shapes in Germany. Additionally, we gather sun radiation data to be used in the estimation process.

2. **Roof Shape Detection**: We train a CNN model to classify roof shapes in the collected images. The detected roof shapes serve as input for the azimuth angle detection step.

3. **Azimuth Angle Detection**: We use CV techniques to determine the azimuth angles of the detected roof shapes. This information is essential for calculating the solar energy potential.

4. **Energy Potential Estimation**: With the roof shape, azimuth angle, sun radiation data, and tilt angle, we estimate the electrical potential and efficiency of solar panels on these roofs.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Pandas
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
