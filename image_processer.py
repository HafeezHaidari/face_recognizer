from PIL import Image
import os
import cv2
import pandas as pd
import numpy as np


def vid_to_pics(vid_path, pics_dir):
  '''
  Extract each frame from a video and put it inside a folder
  '''
  vid_path = vid_path
  pics_dir = pics_dir
  os.makedirs(pics_dir, exist_ok=True)
  video = cv2.VideoCapture(vid_path)
  frame_count = 0
  while True:
    ret, frame = video.read()
    if not ret:
      break
    frame_name = os.path.join(pics_dir, f'frame_{frame_count}.jpg')
    cv2.imwrite(os.path.join(pics_dir, f'{frame_count}.jpg'), frame)
    frame_count += 1
  video.release()
  print(f'Extracted {frame_count} frames from {vid_path} to {pics_dir}.')



def convert_and_resize_image(image_path, output_path, size):
  """
  Convert an image to grayscale and resize it to a given resolution.
  """
  image = Image.open(image_path)
  grayscale_image = image.convert('L')
  resized_image = grayscale_image.resize(size)
  resized_image.save(output_path)



def process_images_in_folder(input_folder, output_folder, size):
  """
  Process all images in a folder: convert to grayscale and resize
  """
  os.makedirs(output_folder, exist_ok=True)
  for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
      input_path = os.path.join(input_folder, filename)
      output_path = os.path.join(output_folder, filename)
      convert_and_resize_image(input_path, output_path, size)


def image_to_flattened_array(image_path):
  '''
  Turn an image into a 1D vector
  '''
  image = Image.open(image_path)
  image_array = np.array(image)
  normalized_image = image_array / 255.0
  flattened_image = normalized_image.flatten()
  return flattened_image


def process_images_in_folder(folder_path):
  '''
  Process all images in a folder: Turn them to 1D vectors and put them into a single dataframe.
  '''
  data = []
  for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', 'webp')):  # Add other image formats if needed
      image_path = os.path.join(folder_path, filename)
      flattened_image = image_to_flattened_array(image_path)
      data.append(flattened_image)

  # Create a DataFrame where each row is the flattened image data
  df = pd.DataFrame(data, columns=[f'pixel{i}' for i in range(len(data[0]))])
  return df
