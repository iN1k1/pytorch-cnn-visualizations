"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
from PIL import Image
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models

from .misc_functions import preprocess_image, recreate_image


class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter,
                 network_mean=(0.485, 0.456, 0.406), network_std=(0.229, 0.224, 0.225),
                 im_size=(224,224,3),
                 save_path='generated', use_gpu=False):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(150, 180, im_size))
        self.use_gpu = use_gpu
        if use_gpu:
            self.model.to("cuda")
        self.network_mean = network_mean
        self.network_std = network_std
        # Create the folder to export images if not exists
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, cudify=self.use_gpu, mean=self.network_mean, std=self.network_std)
        # Define optimizer for the image
        optimizer = Adam([self.processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.cpu(), mean=self.network_mean, std=self.network_std)
            # Save image
            if i % 5 == 0:
                Image.fromarray(self.created_image).save( os.path.join(self.save_path, 'layer_vis_l' + str(self.selected_layer) +
                                 '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'))

    def visualise_layer_without_hooks(self, save_iter=5, num_iter=50):
        # Process image and return variable
        self.processed_image = preprocess_image(self.created_image, resize_im=False, cudify=self.use_gpu, mean=self.network_mean, std=self.network_std)
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas later layers need less
        optimizer = SGD([self.processed_image], lr=5, weight_decay=1e-6)
        for i in range(1, num_iter+1):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = self.processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                x = layer(x)
                if index == self.selected_layer:
                    # Only need to forward until the selected layer is reached
                    # Now, x is the output of the selected layer
                    break
            # Here, we get the specific filter from the output of the convolution operation
            # x is a tensor of shape 1x512x28x28.(For layer 17)
            # So there are 512 unique filter outputs
            # Following line selects a filter from 512 filters so self.conv_output will become
            # a tensor of shape 28x28
            self.conv_output = x[0, self.selected_filter]
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = torch.mean(self.conv_output.to("cpu"))
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.item()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image.to("cpu"), mean=self.network_mean, std=self.network_std)
            # Save image
            if i % save_iter == 0:
                Image.fromarray(self.created_image).save(os.path.join(self.save_path, 'layer_vis_l' + str(self.selected_layer) +
                                         '_f' + str(self.selected_filter) + '_iter' + str(i) + '.jpg'))


if __name__ == '__main__':
    cnn_layer = 2
    filter_pos = 1
    # Fully connected layer is not needed
    pretrained_model = models.vgg16(pretrained=True).features
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
