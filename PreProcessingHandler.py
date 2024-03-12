import cv2
import os


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
                img = cv2.imread(filepath)
                resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(filepath, resized_img)
