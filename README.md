# Style-Transfer-On-Images

#### The Fault in our stars image stylized with abstract flaming style art
![Stylization Sample 1](https://github.com/AILucifer99/Style-Transfer-On-Images/blob/main/Results/comp-1.jpg?raw=true)

#### The Fault in our stars stylized with Vincent Van Gogh Style art
![Stylization Sample 2](https://github.com/AILucifer99/Style-Transfer-On-Images/blob/main/Results/comp-2.jpg?raw=true)


An implementation of Advanced Neural Style Transfer Algorithm using Pytorch.
This project demonstrates neural style transfer using deep learning techniques. Neural style transfer is a fascinating concept that combines the style of one image (typically an artwork) with the content of another image to create a unique artistic output.

# Overview
Neural style transfer utilizes a pre-trained convolutional neural network (CNN) to extract features from both the style image and the content image. 
By minimizing the difference between the extracted features of the content image and the generated image, while also matching the style features of the style image, 
we can create a visually appealing image that combines the style and content. This project provides a Python implementation of neural style transfer using the popular deep learning library, Pytorch.

# Setup
Before running the project, make sure you have the following dependencies installed:

1.   Python 3 (version 3.6 or higher)
2.   Pytorch (version 1.4 or higher)
3.   NumPy
4.   Matplotlib
5.   PIL (Python Imaging Library)
6.   OpenCV

# Installing the Library for local usage
Run the command, `pip install -q torch numpy matplotlib pillow opencv-python`

# Usage from the scratch
1.   Firstly clone the repo using the command, `git clone https://github.com/AILucifer99/Style-Transfer-On-Images`
2.   Perform a directory change using the command, `cd Style-Transfer-On-Images`
3.   Open the script named as `main_stylize.py`
4.   Place the content images in the directory, `data/original-images/<name of the original image>.<extension>`
5.   Place the stylize images in the directory, `data/style-images/<name of the style image>.<extension>`
6.   Now just save the python file and run the command, `python main_stylize.py`

# Running on the pipeline on Google Colab
1.   Just download the `Neural-Style-Transfer.ipynb` file and upload it in the respective google drive.
2.   Open the ipynb file using google colab. 
3.   Create a folder named as `data`
4.   Inside the `data` folder create two other folders named as `original-images` and `style-images`
5.   Upload the respective images to the folders and replace the following :- `original_image = load_image("data/original-images/sample.jpg")` and 
`style_image = load_image("data/style-images/style.jpg")` respectively. 
6.   Once the generation is done, the results will be saved in a folder created dynamically during the experimentation, named as `generated-images-<random_number>`
