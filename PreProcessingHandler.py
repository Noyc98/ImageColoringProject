import cv2
import os
import numpy as np
import torch


class PreProcessing:
    def __init__(self):
        pass

    def convert_folder_to_grayscale(self, input_folder, output_folder):
        # Iterate over each file in the input folder
        for filename in os.listdir(input_folder):
            # Check if the file is a JPEG image
            if filename.endswith(".jpg"):
                # Read the image
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                # Convert the image to grayscale
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Save the grayscale image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, grayscale_image)

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

    def resize_images(self, folder, target_size):
        for filename in os.listdir(folder):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                filepath = os.path.join(folder, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
                resized_img = cv2.resize(img, target_size[::-1], interpolation=cv2.INTER_AREA)  # Resize the grayscale image
                resized_img = np.expand_dims(resized_img, axis=-1)  # Add a single channel dimension
                cv2.imwrite(filepath, resized_img.astype(np.uint8))


    def pad_image(self, image, pad_size):
        return cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

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

    def extend_dataSet_laplacian(self, input_directory, output_directory, num_levels=4):
        # Iterate over each file in the input folder
        for filename in os.listdir(input_directory):
            # Check if the file is a JPEG image
            if filename.endswith(".jpg") or filename.endswith('.jpeg'):
                # Read the image
                image_path = os.path.join(input_directory, filename)
                image = cv2.imread(image_path)

                # Pad the image
                pad_size = 2 ** num_levels
                padded_image = self.pad_image(image, pad_size)

                # Build Laplacian pyramid
                laplacian_pyramid = self.build_laplacian_pyramid(padded_image, num_levels)

                # Save the pyramid levels
                output_filename = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}_level_4.jpg')

                # Save the grayscale image to the output folder
                # output_path = os.path.join(output_directory, output_filename)
                cv2.imwrite(output_filename, laplacian_pyramid[num_levels])
