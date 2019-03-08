## self-steering-ue4-beta
A deep learning project to generate training data for image segmentation and steering prediction using the Unreal 4 Game Engine

I've been studying deep learning for a few months now. Earlier this year I came across fast.ai. Their approach and philosphy really resonated with me and I've enjoyed using their libraries of great Python tools. Lesson 3 of their online course deals with image semgentation and image regression. The lesson mentions how tedious it is to create training data for image segmentation, often requiring a human to create masks and labels for each datapoint by hand.

I come from a video game art background. I've been making 3d digital environments for games for over 20 years, so I thought that I could devise a method to generate training data from a game engine, specifically, Unreal 4, using the fast.ai libraries and Pytorch to train a neural network that would predict the image segmentation of a camera shot taken from a 3D environment in the game engine. For further practice, I would then use those segmented images and steering data captured from the game engine as the training data for another neural network that would make steering predictions to keep a vehicle on a path.

# Requirements

- Python 3.7+
- NumPy
- Pandas
- Pyperclip
- fastai 1.0.46

# Descriptions of contents

- SteeringValuesProcessing.ipynb - jupyter notebook containing cells intended to be run individually as needed. These helped with the data management of steering values captured from Unreal, processing screenshots for size, name, and format needed for NN training.
- steerPred_imageSegUE4.ipynb - juyter notebook that uses transfer learning from resnet34, data collected from Unreal and the SteeringValueProcessing notebook to train a Unet neural network to make image segmentation predictions.
- steerPred_steerPredict.ipynb - notebook that uses transfer learning from resnet34, data generated from Unreal and the SteeringValueProcessing notebook to train a convoluted neural network to make steering predictions for a vehicle traveling along a path.
- startPredictors.py - script that runs in the background that makes steering predictions based on a camera output in Unreal 4

# Generating a dataset for image segmentation

In Unreal I created a simple path that winds through a little village. The path is clear, mostly uniform, gently curving, with no forks or intersections. I then created a camera with very simple controls: it always moves forward at a constant speed. Steering is controlled through user input (keyoard or game controller). At every frame I recorded the steering value coming from the user into an array. After completion the array is dumped to a text file, which I process in the SteeringValuesProcessing notebook into a csv, which Unreal can read in. So, esentially what I've done is record the path of the camera. When I read that csv back into an array and step through each value every frame, then feed that into the steering input, the camera will follow the exact same path as I initially drove, as long as the intial start position and orientation aren't changed. I then captured a screenshot at every frame while the camera followed the path. These screenshots comprise the images of the dataset. To create the labels I duplicated the 3D environemnt, replaced all the materials with flat, unlit, brightly-colored materials that correstpond to the different classes I want to predict; i.e road, grass, fence, house, sky, etc. Using the same camera moving along the recorded path, I then took screenshots at every frame. The result is a perfectly aligned label in seconds.

<p align="center">
<img src="media/image.png" width="600px"><br>
</p>

<h2 align="center">
<b>Image</b>
</h2>

<p align="center">
<img src="media/label.png" width="600px"><br>
</p>

<h2 align="center">
<b>Label</b> 
</h2>
