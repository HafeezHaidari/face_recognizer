'''
Import all necessary libraries
'''
import cv2
import pickle
import numpy as np
from face_recognizer import NeuralNetwork


def initialize():
    '''
    Initialize the neural network and load weights and biases from the .pkl file
    '''
    neural_network = NeuralNetwork(128*128, 8192, 4096, 1024, 512, 1)
    neural_network.load_model('test_model.pkl')
    return neural_network


def recognize_face(nn, image):
    '''
    Define the function recognize_face that takes the neural network and an image as input and returns the probability of the image being a face
    '''
    preprocess_image = image
    output = nn.forward_prop(preprocess_image)
    prediction = nn.get_predictions()
    probability = output[0,0]
    
    return probability


def main():
    '''
    Main function, intitializes the neural network, loads the face cascade and starts the video capture
    '''
    # Initialize the neural network and the model for face detection
    neural_network = initialize()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start the video capture
    video_capture = cv2.VideoCapture(0)
    expansion_factor = 0.2

    while True:
        ret, frame = video_capture.read()
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            x_expanded = int(x - w* expansion_factor / 2)
            y_expanded = int(y - h* expansion_factor / 2)
            w_expanded = int(w * (1 + expansion_factor))
            h_expanded = int(h * (1 + expansion_factor))
            
            x_expanded = max(0, x_expanded)
            y_expanded = max(0, y_expanded)
            w_expanded = min(frame.shape[1] - x_expanded, w_expanded)
            h_expanded = min(frame.shape[0] - y_expanded, h_expanded)
            
            # Rescale the image to 128x128 pixels
            face_resized = cv2.resize(frame, (128, 128))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
            
            # Normalize the image
            face_normalized = face_gray / 255.0  
            
            # Reshape the image to a 1D vector
            face_reshaped = np.reshape(face_normalized, (128*128, 1)) 
            
            # Try and recognize the face
            probability = recognize_face(neural_network, face_reshaped)
            
            # If probability is greater than 0.5, the face is recognized and the rectangle is green, otherwise it is red
            if probability > 0.5:
                color = (0, 255, 0) 
            else:
                color = (0, 0, 255)
            # Add the probability to the label
            label = f'{probability * 100:.2f}% match'
            
            # Display the rectangle and the label
            cv2.rectangle(frame, (x_expanded, y_expanded), (x_expanded + w_expanded, y_expanded + h_expanded), color, 2)
            cv2.putText(frame, label, (x_expanded, y_expanded + h_expanded + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            face_image = frame[y_expanded:y_expanded+h_expanded, x_expanded:x_expanded+w_expanded]
        
        # Show the video
        cv2.imshow('Video', frame)
        
        # If the user presses 'q' or closes the window, the video capture is stopped
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows() 
    

# Run the main function
if __name__ == '__main__':
    main()
