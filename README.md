# Indian-Traffic-Sign-Detection

ADAM Optimizer

## Introduction:
Traffic sign detection is a crucial task in the field of autonomous vehicles and intelligent transportation systems. In this project, we focus on Indian traffic sign detection using Convolutional Neural Networks (CNNs). To achieve this, we leverage the power of transfer learning by employing the InceptionV3 prebuilt model. Additionally, we utilize ImageDataGenerator for image augmentation to enhance the robustness of our model. The ADAM optimizer is chosen to efficiently optimize the CNN's parameters during training. The dataset used for training and evaluation is sourced from Kaggle, containing labeled images of various Indian traffic signs.

### 1. Data Collection and Preparation:
The dataset from Kaggle comprises images of different Indian traffic signs, each labeled with the corresponding class information. These images are divided into training, validation, and test sets. Before feeding the data into the model, we preprocess the images by resizing them to a consistent input shape, typically expected by the InceptionV3 model, and normalize the pixel values to a range suitable for neural network training .

- Kaggle Datset Link: https://www.kaggle.com/code/aryashah2k/indian-traffic-sign-classification/input

### 2. Transfer Learning with InceptionV3:
InceptionV3 is a powerful prebuilt CNN architecture that has been trained on a massive dataset and has learned to extract highly abstract features from images. By employing transfer learning, we can take advantage of these learned features and fine-tune the model for our specific task, which helps save time and computational resources. We remove the last few layers of the InceptionV3 model and replace them with custom layers to adapt the model for traffic sign detection.

### 3. Image Augmentation using ImageDataGenerator:
ImageDataGenerator is a utility provided by Keras that allows us to apply various transformations to the training images on-the-fly during training. This helps to augment the dataset and improve the model's ability to generalize to unseen data. Common augmentations include rotations, flips, shifts, and zooms, which introduce diversity to the training data without the need for manually creating augmented images.

### 4. ADAM Optimizer:
The ADAM (Adaptive Moment Estimation) optimizer is a popular choice for deep learning tasks. It combines the benefits of both AdaGrad and RMSprop optimizers and efficiently adapts the learning rates for each parameter during training. ADAM typically converges faster and more effectively than traditional stochastic gradient descent (SGD), making it well-suited for training complex models like CNNs.

### 5. Model Training:
During the training process, the model is fed with augmented training data in batches. The ADAM optimizer updates the model's weights based on the gradients calculated from the loss function. The model is trained for several epochs, and the learning rate, batch size, and other hyperparameters are carefully tuned to achieve optimal results.

### 6. Evaluation on Test Set:
Once the training is complete, the model is evaluated on the test set to measure its performance on unseen data. Common evaluation metrics include accuracy, precision, recall, and F1-score. The model's ability to correctly classify traffic signs in real-world scenarios is assessed through this evaluation.

### 7. Prediction on Unseen Images:
To further validate the model's generalization, we use some unseen images not present in the training or test sets. The model makes predictions on these images, and the results are visually inspected to verify the model's capability to detect Indian traffic signs accurately.

## Conclusion:
Indian Traffic Sign Detection using InceptionV3 CNN with ImageDataGenerator and ADAM Optimizer is an effective approach for traffic sign detection tasks. By leveraging transfer learning with a prebuilt model, employing data augmentation, and using an efficient optimizer, the model can achieve good accuracy even with a relatively small amount of data. This technology holds great potential for enhancing the safety and efficiency of transportation systems in India and beyond. However, continuous improvement and optimization are necessary as new data becomes available and hardware capabilities advance.
