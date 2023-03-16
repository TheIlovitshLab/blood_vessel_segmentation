# Blood Vessels Segmentation
This repository contains the code for blood vessel segmentation in microscope images,
along with instructions for running both segmentation and training on new datasets.
<img src="resources/io_rg11_3_striatum.png">

## Segment New Images
To segment new microscope images, follow these steps:
1. Copy the microscope images to the **'dataset/data_for_segmentation'** folder,
ensuring that the folder only contains **'.tiff'** files.
2. Run the following command in the terminal:
>python predict --model <model_name>.

This will generate predictions for the images, which will be saved in **'dataset/data_for_segmentation'**.
3. Run the Matlab file:
>auto_seg

to further process the predictions.
4. Finally, open EBreaverApp and continue with the usual workflow.



## Training
To train a new model on your own dataset, follow these steps:

### Date Preparation

1. Create the following folders in dataset: **'raw_data'**, and **'train'**.
2. Copy the folders to be processed (for example: **'RG11 - MB 175'**) to **'dataset/raw_data'**.
These folders should contain the original images (**'.tiff)'**, the masks (**'.mat'**), and
User Verified Table (**'.mat'**).
3. Run the Matlab file:
>extract_data.
 
This will create two new folders in **'train'**: **'images'** and **'masks'**.


### Train
Run the following command in the terminal:
> python main.py --train True --epochs <num_epochs> --batch_size <batch_size>

This will train the model, and after training is finished, the models and
training metrics plots will be saved in **"outputs/<training_name>/'**.

### Test
Run the following command in the terminal:
>python main.py --test True --model <model_name>.

This will test the model, and the results will be saved in **'outputs/<training_name>/test_plots/'**.


