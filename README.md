# Neural Style Transfer with Flask

## Overview

This project implements Neural Style Transfer (NST) using Convolutional Neural Networks (CNNs) to merge the content of one image with the style of another. The application is built using Python and Flask, allowing users to upload images, apply the neural style transfer algorithm, and view the generated artistic image.

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Neural Style Transfer Explained](#neural-style-transfer-explained)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Project Structure

```plaintext
├── src
│   ├── deprocess.py        # Deprocessing utilities for images
│   ├── loss.py             # Loss functions used in style transfer
│   ├── preprocess.py       # Preprocessing utilities for images
├── utils
│   └── common_utils.py     # Common utilities used across the project
├── templates
│   └── index.html          # Main HTML page for the Flask app
├── static
│   ├── styles
│   │   └── mainpage.css    # CSS styles for the web page
│   └── output.jpg          # Placeholder for the generated image
├── main.py                 # Flask application and core logic
└── requirements.txt        # Python dependencies
```


## Features
- **Image Upload:** Users can upload a content image and a style image through a simple web interface.
- **Neural Style Transfer:** The core algorithm applies the style from the style image onto the content image.
- **Real-time Result Display:** Once the process is complete, the generated image is displayed on the same page.
- **Logging:** Application logs are stored in a MySQL database for easy tracking and monitoring.

## Neural Style Transfer Explained
Neural Style Transfer (NST) is a deep learning technique that synthesizes an image that combines the content of one image with the style of another. The algorithm uses the following key steps:

1. **Content Representation:** Extract the content features from a content image using a pre-trained CNN (such as VGG-19).
2. **Style Representation:** Extract the style features from a style image, typically focusing on textures and patterns.
3. **Optimization:** Minimize the content loss and style loss to iteratively modify a generated image so that it matches the content of the content image and the style of the style image.

## Setup Instructions

### Prerequisites
- Python 3.8+
- NVIDIA GPU (optional, for faster training)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/neural-style-transfer-flask.git
   cd neural-style-transfer-flask

2. **Create a Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate 

3. **Install the dependencies**
   ```bash
   pip install -r requirements.txt

4. **Create a MYSQL Database on local server**

5. **Update the following on the common_utils.py file**
   ```bash
   host='',
    database='',
    user='',
    password=''

6. **Run Flask App**
   ```bash
   python main.py
   

### Dependencies
- Flask: Web framework for building the application.
- TensorFlow: Used for the deep learning model and neural style transfer algorithm.
- Pillow: For image processing tasks.
- MySQL: Database for logging purposes.
- SQLAlchemy: ORM for managing the database.
