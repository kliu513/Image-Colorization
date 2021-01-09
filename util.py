import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import color
from torchvision.utils import make_grid
from torchvision import transforms, datasets, models

def show_tensor_images(image_tensor, num_images, size=(1, 32, 32)):
    
    # Function for visualizing images: Given a tensor of images, number of images, and
    # size per image, plots and prints the images in an uniform grid.
 
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
def show_tensor_lab(image_tensor, num_images, size=(1, 32, 32)):

    # Function for visualizing images: Given a tensor of images, number of images, and
    # size per image, plots and prints the images in an uniform grid.
    
    image_unflat = image_tensor.detach().cpu().numpy()
    img = [color.lab2rgb(np.transpose(image, (1,2,0))) for image in image_unflat[:num_images]]
    image_grid = make_grid(torch.tensor(np.transpose(img, (0,3,1,2))), nrow=4)
    plt.imshow(image_grid.permute(1,2,0))
    plt.axis('off')
    plt.show()

# Function for cropping an image tensor: Given an image tensor and the new shape,
# crops to the center pixels.
def crop(image, new_shape):
    # Parameters:
    # image: torch.tensor, image tensor of shape (batch size, channels, height, width)
    # new_shape: torch.Size object, the expected shape of x
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - round(new_shape[2] / 2)
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - round(new_shape[3] / 2)
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]
    return cropped_image
   
    