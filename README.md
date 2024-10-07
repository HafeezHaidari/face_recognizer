# face_recognizer
Face Detector and Recognizer in Python using raw python.

### Additional libraries
Numpy, Pandas, Opencv-python, pickle

### Necessities
Your own video that the model can be trained on.

After using image_processing.py on your video containing your face you need to add a 'label' column to the dataframe with the value 1 so that the neural network knows that this face needs to be recognized.
When running a video that does not contain your face you again add a 'label' column but this time it has the value 0.

Running the face_recognizer.py initializes the model, trains it and then saves it as a .pkl file for further use.
