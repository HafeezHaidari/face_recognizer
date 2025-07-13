# Face Recognition Neural Network App

A Python application for face recognition using a custom-built neural network and OpenCV for real-time video processing.

## Features

* **Data Loading**: Efficiently loads and processes large image datasets from CSV files using pandas.
* **Image Processing**: Converts video frames to flattened image vectors and prepares data for the network.
* **Neural Network**: Implements a multi-layer neural network from scratch with configurable architecture, forward/backward propagation, and model persistence via pickle.
* **Real-Time Recognition**: Captures webcam video, detects faces, and performs live recognition, displaying results in a GUI window.

## Prerequisites

* Python 3.7+
* pip (Python package manager)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/face-recognition-app.git
   cd face-recognition-app
   ```
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

## Usage

### 1. Prepare Your Data

* Use the `image_processer.py` module to extract frames from a video:

  ```bash
  python -c "from image_processer import vid_to_pics; vid_to_pics('input_video.mp4', 'frames/')"
  ```
* Convert a folder of images into a DataFrame for training:

  ```python
  from image_processer import process_images_in_folder

  df = process_images_in_folder('frames/')
  df.to_csv('complete_image_data.csv', index=False)
  ```

### 2. Train the Model

* Edit `face_recognizer.py` to adjust network architecture or training parameters.
* Run the training script:

  ```bash
  python face_recognizer.py
  ```
* A trained model will be saved as `test_model.pkl`.

### 3. Real-Time Face Recognition

* Launch the main application:

  ```bash
  python main.py
  ```
* A window will open showing webcam input with recognized faces labeled. Press `q` or close the window to exit.

## Project Structure

```plaintext
├── data_loader.py         # Loads CSV data and splits into training/dev sets
├── face_recognizer.py     # Defines and trains the neural network classifier
├── image_processer.py     # Video-to-frame conversion and image flattening
├── main.py                # Real-time face recognition demo using OpenCV
├── requirements.txt       # List of Python dependencies
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new feature branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m 'Add YourFeature'`.
4. Push to the branch: `git push origin feature/YourFeature`.
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
