The project involves training a model for land-use classification from satellite images using a given dataset. The dataset is a multiclass land-use dataset, and the goal is to predict the dominant land-use category in a given satellite image. The dataset comes from a paper titled "Semantic Segmentation of Satellite Images for Land-Use Classification" (https://ieeexplore.ieee.org/abstract/document/8519248).

## Task 1: Single-label Prediction

Data Preparation:

Split the data into training, validation, and testing sets using a random generator seed.
Verify that the splits are disjoint and set up the code to run when the split is re-generated.
Write a dataset class for the train/val/test datasets without using pre-built dataset classes from TensorFlow or PyTorch.
Training:

Train a model using finetuning.
Implement custom data augmentation schemes on the validation set.
Evaluate the model on the validation and test sets.
Report average precision, accuracy per class, mean average precision, and loss curves for both training and validation sets.
Testing:

Report test performance after selecting hyperparameters on the validation set.

## Results for task1:

![image](https://github.com/samuelgjy/multiclass-landuse/assets/110824653/9350d53f-7ba2-4fca-82c7-b16b85e6e062)


## Task 2: Multi-label Prediction

Data Preparation:

Modify labels for multi-label prediction.
Set the label "Annualcrop" to 1 whenever "PermanentCrop" is 1, and vice versa.
Set "HerbaceousVegetation" to 1 if "Forest" is 1 (but not the other way round).
Training:

Finetune a new model with the modified label schema.
Implement one data augmentation schema for training.
Report the same metrics and curves as in Task 1.

## Results for task2:
![image](https://github.com/samuelgjy/multiclass-landuse/assets/110824653/d9b35732-5d23-429f-b6a3-2ffb6fd9c608)
