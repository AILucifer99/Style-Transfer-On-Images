import cv2
import os
from tqdm import tqdm as tqdm


# For experimentation only.
# Function to generate an interpolation video from the images generated during the optimization
def interpolation_video_generation(image_folder, output_path, num_frames=30, interpolation_type='linear', fps=30, resize=None):
    """
    Generates an interpolation video from a sequence of images in a folder.

    Args:
        image_folder (str): Path to the folder containing the input images.
        output_path (str): Path to save the generated video.
        num_frames (int, optional): Number of frames in the interpolation video.
            Defaults to 30.
        interpolation_type (str, optional): Interpolation type.
            'linear' for linear interpolation or 'cubic' for cubic interpolation.
            Defaults to 'linear'.
        fps (int, optional): Frames per second of the output video.
            Defaults to 30.
        resize (tuple, optional): Desired dimensions (width, height) for resizing the images.
            Defaults to None (no resizing).
    """

    # Get the list of image file names in the folder
    image_files = sorted(os.listdir(image_folder))
    num_images = len(image_files)

    # Check if the number of images is sufficient for interpolation
    if num_images < 2:
        print("Error: Insufficient number of images for interpolation.")
        return

    # Determine the interpolation step based on the number of frames
    step = (num_images - 1) / (num_frames - 1)

    # Create an array to store the interpolated frames
    interpolated_frames = []

    # Perform interpolation between consecutive images
    for i in tqdm(range(num_frames)) :
        # Calculate the indices of the two nearest images for interpolation
        idx1 = int(i * step)
        idx2 = min(idx1 + 1, num_images - 1)

        # Load the two nearest images
        img1 = cv2.imread(os.path.join(image_folder, image_files[idx1]))
        img2 = cv2.imread(os.path.join(image_folder, image_files[idx2]))

        # Resize the images if specified
        if resize is not None:
            img1 = cv2.resize(img1, resize)
            img2 = cv2.resize(img2, resize)

        # Perform interpolation based on the interpolation type
        if interpolation_type == 'linear':
            weight = i * step - idx1
            interpolated_frame = cv2.addWeighted(img1, 1 - weight, img2, weight, 0)
        elif interpolation_type == 'cubic':
            interpolated_frame = cv2.resize(img1, img2.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)

        interpolated_frames.append(interpolated_frame)

    # Save the interpolated frames as a video
    height, width, _ = interpolated_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in interpolated_frames:
        video_writer.write(frame)
    video_writer.release()

    print(f"Interpolation video saved at {output_path}")


if __name__ == "__main__" :
    image_folder = f"path of the images"
    output_path = "interpolation_video.mp4"
    num_frames = 500
    interpolation_type = "cubic"
    fps = 120
    resize = (1200, 640)

    interpolation_video_generation(
        image_folder, output_path, num_frames, 
        interpolation_type, fps, resize
        )
