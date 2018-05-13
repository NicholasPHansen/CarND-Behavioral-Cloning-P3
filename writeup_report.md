# **Behavioral Cloning**


Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---

## Model Architecture and Training Strategy

### Choosing a model

For this project I decided to implement the Nvidia end-to-end model.
This model was already designed for the purpose of teaching a neural network to drive a car.
This model cosists of a five convolutions layers, followed by four fully connected layers.

The final model was implemented in the `nvidia_model()` method (model.py line 83).
The base Nvidia model was tweaked to have the following:

- A input normalization layer
- Cropping layer
- Single output

The below table gives a more detailed description of each of the layers.

| Layer                 |     Description                               |
|:----------------------|----------------------------------------------:|
| Input                 | 160x320x3 RGB image                           |
| Input Normalization   | Normalize the pixel values to have 0 mean     |
| Image cropping        | Crop out the top and bottom of the images     |
| Convolution 24x5x5    | 2x2 stride, with RELU activation function     |
| Convolution 36x5x5    | 2x2 stride, with RELU activation function     |
| Convolution 48x5x5    | 2x2 stride, with RELU activation function     |
| Convolution 64x3x3    | 1x1 stride, with RELU activation function     |
| Convolution 64x3x3    | 1x1 stride, with RELU activation function     |
| Flattening            |                                               |
| Fully connected       | 1164x1 fully connected layer                  |
| Fully connected       |  100x1 fully connected layer                  |
| Fully connected       |   50x1 fully connected layer                  |
| Fully connected       |   10x1 fully connected layer                  |
| Output                |    1x1 fully connected layer                  |

### Training

To train the model I used a larger batch size of 128 images.
I decided to train my model for 3 epochs over the data, as this number of epochs showed good validation loss, which did not really improve by training for more epochs.

The loss function was chosen as a mean-squared-error (mse) function, with and Adam optimizer, which meant that learning rate would be automatically updated during training.

To validate the model and avoid overfitting, I split the used dataset in to a 80-20, with the model training on 80% of the data, and testing on the remaining 20%.


### Data collection

I collected a few datasets for this project.
The first dataset consisted of:

- A single center driving lap on the first track.

The first data set would be appended by adding the following:

- Several recovery clips, from different parts of the first track
- Short clips of turning from some of the corners with sand to one side

The second dataset that was collected contained the following:

- 3 laps around the first track
- A single lap around the second track

#### First data set

Using the base Nvidia model (without any modifications), I did not encounter any problems with overfitting, when looking at the loss functions of the training and test sets.

I did however see poor performance when testing the trained model on the simulator track.
The model would keep steering left, and start to go in a circle.

There are a few things to consider about this model and dataset:

- The model would use all three images, and use steering corrections for the left and right images
- The dataset contained a single lap around the first track, and nothing else.
- The model would use all the pixels in the image

The above described behaviour indicated to me the following: the model had only learned to turn left, as that was all it had ever seen me do, and it did not matter what the camera images saw (sand, grass, plamtrees, water etc.), the steering angle would be giving a left turn (except one turn on the track mind you).
Also the model would be trained on the entire image from the cameras, which contained alot of sky/hills/trees in the upper part of the image, which has nothing to do with the outcome of the steering angle.

A quick solution to try and combat the problem was to append the flipped images and steering angles, and crop the images.
Using this approach the model would perform better, and would now start to move along the track, but would eventually fall of, and then fail miserably from there on.

##### Appending the dataset

I noticed that the model behaved poorly when reaching the sidewalls or edges of the driving lane.
During the data collection, I had driven (to the best of my ability), in the center of the track, giving the model no info of how to handle the case of being at the edge of the track.

I appended some recovery clips to the dataset, from different parts of the track, focusing on the parts where the track had a different look (e.g. the parts of the track where one side is covered in sand).
Again, I also added the inverted image/steering angles.

Using this data, the model would finally start to show a promising behaviour, and would start to drive around the track.
However, the model would consitently fail when reaching the first sharp turn, and would drive into the sand parts.

This left me with some options:

- Give the model more data of the specific turn
- Try another dataset

I opted for the second options, and will explain why in the following section.

#### Second Dataset

I started to notice that the model would have a predictive behaviour, and would behave consitently according to the part of the track that it would be driving on.
This behaviour included, veering to the right when entering the bridge, and going all the way to the edge (and up on the curb) when taking some of the sharper corners, both of which were undesired behaviour.
This indicated to me, that I had unintentionally thaught the model some bad behaviour, by recording this behaviour in the data provided to the model.

To combat this, I decided to record a completly new dataset which would avoid the above described undesired behaviour.
When collecting data for this dataset, I decided to do the following:

- Drive several laps around the first track
- Drive a single lap around the second track
- I would not record specific parts of the track
- I would not record specific recovery runs

During the capture on both tracks, I tried to not drive without giving special attention to "how" I was driving.
In other words, I would drive as if I was to drive around the track in a "racing game", and not teach someone else how to do it.
Also, by training on the second track, where the terrain and track would vary quite alot, was something I hoped would be benefitial for creating a more generalized model.

This dataset would contain 4515 rows in the `.csv` file, which would yield 27090 images using the generator to train the network on.

After training the network, I get the following computed losses:

| Epoch  | Training loss | Validation Loss |
|:------:|:-------------:|:---------------:|
| 1      | 0.0921        | 0.0727          |
| 2      | 0.0729        | 0.0702          |
| 3      | 0.0694        | 0.0648          |

---

## Results

The final model is stored in the `model.h5` file, and can be loaded from there.
When testing the model it would successfully drive a lap around the first track, which is recorded in the `run.mp4` video file.
The video can also be viewed on YouTube [here](https://youtu.be/XJ7uutpkoV4).