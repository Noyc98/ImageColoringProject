from PreProcessingHandler import PreProcessing


def main():
    pre_processing = PreProcessing()
    # pre_processing.convert_folder_to_grayscale("flowers_color", "flowers_gray")
    
    max_width, max_height = pre_processing.find_largest_image_size("flowers_gray")
    target_size = (max_width, max_height)
    # pre_processing.resize_images("flowers_gray", target_size)
    pre_processing.extend_dataSet_laplacian("flowers_gray","extended_dataSet",4)
    pre_processing.resize_images("extended_dataSet", target_size)
    # pre_processing.resize_and_replace_images("extended_dataSet","extended_dataSet", max_width, max_height)

main()