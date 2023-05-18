# VGG Network Neural Style Transfer

# The layers - conv 1-1 || conv 2-1 || conv 3-1 || conv 4-1 || conv 5-1

# Total loss can be calculated as 
# The combination of the alpha times the loss of the original image and the generated image

# We have to consider the following convolutional layers outputs
# The layers will be acting as the feature extractor for the content image andthe style images

# 0    -  conv 1-1
# 5    -  conv 2-1
# 10   -  conv 3-1
# 19   -  conv 4-1
# 28   -  conv 5-1

# Libraries required for the pipeline to work.

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import os
import random
import math
from PIL import Image
from tqdm import tqdm as tqdm
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

# Loading the pre-trained VGG model for the feature extractor
model = models.vgg19(pretrained=True).features

# To see the model architecture uncomment the next line and run the cell.
# print("The Model architecture is provided below :- {}".format(model))

# Using the VGG Model as the feature extractor for the generation pipeline
class NSTVGG(nn.Module) :
    def __init__(self, features) :
        super(NSTVGG, self).__init__()
        self.chosen_features = features
        self.model = models.vgg19(pretrained=True).features[:33]

    def forward(self, x) :
        features = []
        for layer_num, layer in enumerate(self.model) :
            x = layer(x)
            if str(layer_num) in self.chosen_features :
                features.append(x)
        return features


def load_image(image_name) :
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)
    

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selected_feature_layer_index = ['0', '5', '10', '19', '28', '30', '32']

original_image = load_image("/content/Fault-1.jpg")
style_image = load_image("/content/flame-1.jpg")

generate_interpolation_video = 1 
image_dimension = [640, 800] # Height and Width respectively.

loader = transforms.Compose(
    [
        transforms.Resize((image_dimension[0], image_dimension[1])),
        transforms.ToTensor(),
    ]
)

# Loading the VGG Model
model = NSTVGG(features=selected_feature_layer_index).to(device).eval()

# Randomized Initialization of the generated image
# generated = torch.randn(original_image.shape, device=device, requires_grad=True)

# Initialization with the original image
generated = original_image.clone().requires_grad_(True)

# System Hyper-parameters
total_steps = 3500
learning_rate = 0.0025

# Content Loss Constant # Previous Value - 1; 1.25; 1.75
alpha = 1

# Style Loss Constant # Previous Value - 0.1; 0.25; 0.45
beta = 0.1

optimizer = optim.Adam([generated], lr=learning_rate)

result_seed = math.ceil(round(int(random.uniform(10, 1000)) + random.uniform(1, 100), 1))

os.makedirs(f"generated-images-{result_seed}", exist_ok=True)

for step in tqdm(range(total_steps)) :

    generated_features = model(generated)
    original_image_features = model(original_image)
    style_image_features = model(style_image)

    style_loss = original_loss = 0
    
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_image_features, style_image_features) :
        
        batch_size, channels, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Calculate the gram matrix for style capturing loss
        G = gen_feature.view(channels, height * width).mm(
            gen_feature.view(channels, height * width).t()
        )

        A = style_feature.view(channels, height * width).mm(
            style_feature.view(channels, height * width).t()
        )

        style_loss += torch.mean((G - A) ** 2)
    
    # Calculation of the total loss so as the apply the otiumizer for the backpropgaration
    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 10 == 0 :
        print("{}Total Loss :- {}".format("\n", total_loss))
        save_image(generated, "/content/generated-images-{}/generated_{}.png".format(
            result_seed, step
            )
        )

if generate_interpolation_video > -1 :
    image_folder = f"/content/generated-images-{result_seed}"
    output_path = "interpolation_video_1.mp4"
    num_frames = 500
    interpolation_type = "cubic"
    fps = 30
    resize = (
        image_dimension[1], 
        image_dimension[0]
        )
    
    interpolation_video_generation(
        image_folder, output_path, num_frames, 
        interpolation_type, fps, resize
        )
