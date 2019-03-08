## self-steering-ue4-beta
A deep learning project to generate training data for image segmentation and steering predictions using the Unreal 4 Game Engine

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
- startPredictors.py - script that runs in the background that makes steering predictions based on camera output in Unreal 4

# Generating a dataset for image segmentation

In Unreal I created a simple path that winds through a little village. The path is clear, mostly uniform, gently curving, with no forks or intersections. I then created a camera with very simple controls: it always moves forward at a constant speed and steering is controlled through user input (keyoard or game controller). At every frame I recorded the steering value coming from the user into an array. After completion the array is dumped to a text file, which I process in the SteeringValuesProcessing notebook into a csv, which Unreal can read in. So, esentially what I've done is record the path of the camera. When I read that csv back into an array in Unreal, step through each value every frame, then feed that into the steering input, the camera will follow the exact same path as I initially drove, as long as the intial start position and orientation aren't changed. I then captured a screenshot at every frame while the camera followed the path. These screenshots comprise the images of the dataset. To create the labels I duplicated the 3D environment, replaced all the materials with flat, unlit, brightly-colored materials that correstpond to the different classes I want to predict; i.e road, grass, fence, house, sky, etc. Using the same camera moving along the previously-recorded path, I then took screenshots at every frame. The result is a perfectly aligned label, generated in seconds.

<p align="center">
<b>Image</b>  <img src="media/image.png" width="300px"><br>
</p>

<p align="center">
<b>Label</b>  <img src="media/label.png" width="300px"><br>
</p>

I then had to do a lot of image processing on the label to get it into a format similar to the dataset I was emulating: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/. This entailed converting the colors to gray values that correspond to a predetermined code. For example, the code for 'house' was 8, so my processing converts all the red pixels in the label into a very dark gray (red 8, green 8, blue 8). The code for 'road' was 3, so my processing converts all the dark blue pixels in the label into a darker gray (red 3, green 3, blue 3). And so on for each of the codes.

# Making image segmentation predictions

The steerPred_imageSegUE4 notebook goes through my process of training a unet. In short, I was able to correctly classify around 95% of the pixels accurately. I thought it was pretty awesome that the unet was so good at figuring out what it was looking at.

<p align="center">
<img src="media/imageSegPredictions.PNG" width="700px"><br>
<b>Dataset image overlayed with label on left. Same dataset image overlayed with prediction on right.</b>

# Generating a dataset for steering prediction (image regression)

Because the labels for the image segmentation are so dark, I multiplied each value by (255/num of classes) so the grayscale image uses up the entire spectrum. It was then a lot easier to see the masks and evaluate predictions. For the targets of the image regression I used the steering values I captured while making the image segmentation dataset. 

# Making steering predictions

The steerPred_steerPredict notebook shows my process of training the cnn. I was never able to get very good loss. It always minimized at  around 0.05, which is not very good for steering values that ranged from -1.0 to 1.0. I pressed on ahead anyway to see how it would perform in the game engine. Later, I found out that this approach had some major flaws, so the current version of this notebook reflects a lot of changes I made to the dataset. See <b>What Went Wrong<b> section below.

# Unreal and predictors setup

I created the startPredictors.py script to take the output of a camera in Unreal, feed that through the image segmentation predictor, then process that prediction and feed it to the steering predictor, which would then in turn feed its prediction back to Unreal. The vehicle (traveling forward at a constant speed, to simplify my life) would update its steering values based on the predictions. For the Unreal setup I used a really simple Blueprint.

<p align="center">
<b>Image</b><img src="media/captureSteering.PNG" width="700px"><br>
</h2>

<p align="center">
<b>Image</b><img src="media/steerBP.PNG" width="700px"><br>
</h2>

For the 2d image capture and clipboard copy functionality I had to use this plugin: https://forums.unrealengine.com/development-discussion/blueprint-visual-scripting/4014-39-rama-s-extra-blueprint-nodes-for-you-as-a-plugin-no-c-required.

# What went wrong:-/

This approach failed. First off, and most obviously, the image segmentation prediction was too slow. It was taking well over a second, which is not enough updates to keep the vehicle on the path, even at very slow speeds. Secondly, my steering predictor was too innacurate. Even after getting the image segmentation the steering prediction was so bad that it was pretty much entirely useless. I had to go back and rethink my approach.

# New Approach

I've decided to abandon image segmentation, for now. I don't know enough about how to make it fast enough for my needs. So instead I decided to focus on the simplest problem: keeping the vehicle on an entirely straight path. In Unreal I created a flat-shaded gray path with bright red lane markings on both sides. My goal was to keep the vehicle between the red lane markers.

I also set my mind to the problem of why my steering predictions were so bad. All of my experimenting and parameter tuning had done nothing to improve it. I reasoned that it must be my training data that was the culprit. Finally, it dawned on me that I was providing contradictory training data to the model. In my effort to provide data that reflected all the situations the vehicle might find itself in, I was inadvertantly providing it with data that was misleading and confusing. For example, I wanted to provide data for what the vehicle should do when it finds itself near the outer edge of the path, so I steered over to the shoulder, then steered back toward the center. I captured the screenshots and steering data for this entire sequence, but actually the initial part of this sequence is not the kind of driving behavior I wanted my vehicle to emulate. When the vehicle is moving directly down the center of the road I don't want it to wildly steer to the shoulder. I want it to continue in the center of the road and make only small steering corrections as needed to keep it there. I realized I would have to be more careful about how I generated my dataset.
