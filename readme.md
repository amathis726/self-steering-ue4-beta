# self-steering-ue4-beta
A deep learning project to generate training data for image segmentation and steering prediction using the Unreal 4 Game Engine

I've been studying deep learning for a few months now. Earlier this year I came across fast.ai. Their approach and philosphy really resonated with me and I've enjoyed using their libraries of great Python tools. Lesson 3 of their online course deals with image semgentation and image regression. The lesson mentions how tedious it is to create training data for image segmentation, often requiring a human to create masks and labels for each datapoint by hand.

I come from a video game art background. I've been making 3d digital environments for games for over 20 years, so I thought that I could devise a method to generate training data from a game engine, specifically, Unreal 4, using the fast.ai libraries and Pytorch to train a neural network that would predict the image segmentation of a camera shot taken from a 3D environment in the game engine. For further practice, I would then use those segmented images and steering data captured from the game engine as the training data for another neural network that would make steering predictions to keep a vehicle on a path.

# Requirements

Python 3.7+
NumPy
Pandas
Pyperclip
fastai 1.0.46

# Descriptions of contents

SteeringValuesProcessing.ipynb - jupyter notebook containing cells intended to be run individually as needed. These helped with the data management of steering values captured from Unreal, processing screenshots for size, name, and format needed for NN training.
steerPred_imageSegUE4.ipynb - juyter notebook that uses transfer learning from resnet34, data collected from Unreal and the SteeringValueProcessing notebook to train a Unet neural network to make image segmentation predictions.
steerPred_steerPredict.ipynb - notebook that uses transfer learning from resnet34, data generated from Unreal and the SteeringValueProcessing notebook to train a convoluted neural network to make steering predictions for a vehicle traveling along a path.
startPredictors.py - script that runs in background that makes steering predictions based on a camera output in Unreal 4
