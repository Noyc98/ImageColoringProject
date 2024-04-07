import cv2 
import os  
import numpy as np  
import torch  

class PreProcessing:
    def __init__(self):
        pass

    # convert images in a folder to grayscale and save them in another folder
    def convert_folder_to_grayscale(self, input_folder, output_folder):
        for filename in os.listdir(input_folder):
            if filename.endswith(".jpg"):  # Check if the file is a JPEG image
                image_path = os.path.join(input_folder, filename)  # Get the full path of the image
                image = cv2.imread(image_path)  # Read the image
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
                output_path = os.path.join(output_folder, filename)  # Define the output path
                cv2.imwrite(output_path, grayscale_image)  # Save the grayscale image

    # find the largest image size in a folder
    def find_largest_image_size(self, folder):
        max_width = 0
        max_height = 0
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                filepath = os.path.join(folder, filename)
                img = cv2.imread(filepath)
                height, width, _ = img.shape
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        return max_width, max_height

    # resize images in a folder to a target size
    def resize_images(self, folder, target_size):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                filepath = os.path.join(folder, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
                resized_img = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_AREA)  # Resize image
                resized_img = np.expand_dims(resized_img, axis=-1)  # Add a single channel dimension
                cv2.imwrite(filepath, resized_img.astype(np.uint8))  # Save resized image

    # pad an image
    def pad_image(self, image, pad_size):
        return cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    # build Laplacian pyramid for an image
    def build_laplacian_pyramid(self, image, levels):
        pyramid = []
        temp_image = image.copy()
        for _ in range(levels):
            blurred = cv2.GaussianBlur(temp_image, (5, 5), 0)
            downsampled = cv2.pyrDown(blurred)
            upsampled = cv2.resize(cv2.pyrUp(downsampled), (temp_image.shape[1], temp_image.shape[0]))
            laplacian = cv2.subtract(temp_image, upsampled)
            pyramid.append(laplacian)
            temp_image = downsampled
        pyramid.append(temp_image)
        return pyramid

    # extend dataset using Laplacian pyramid decomposition
    def extend_dataSet_laplacian(self, input_directory, output_directory, num_levels=4):
        for filename in os.listdir(input_directory):
            if filename.endswith(".jpg") or filename.endswith('.jpeg'):
                image_path = os.path.join(input_directory, filename)
                image = cv2.imread(image_path)
                pad_size = 2 ** num_levels
                padded_image = self.pad_image(image, pad_size)
                laplacian_pyramid = self.build_laplacian_pyramid(padded_image, num_levels)
                output_filename = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}_level_4.jpg')
                cv2.imwrite(output_filename, laplacian_pyramid[num_levels])
