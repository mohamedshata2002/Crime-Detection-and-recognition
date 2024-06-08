import cv2

def is_low_light(image, threshold=100):
    """
    Check if the image is low-light based on average pixel intensity.
    
    Args:
        image (numpy.ndarray): The input image as a numpy array.
        threshold (int): Threshold value for average pixel intensity.
    
    Returns:
        bool: True if the image is low-light, False otherwise.
    """
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average pixel intensity
    avg_intensity = cv2.mean(img_gray)[0]
    
    # Check if average intensity falls below the threshold
    return avg_intensity < threshold

# Example usage:
# Read image from file
# image = cv2.imread('image.jpg')

# Or, capture image from camera
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()
# image = frame.copy()
# cap.release()

# Check if the image is low-light
# low_light = is_low_light(image)
# print("Low light:", low_light)
