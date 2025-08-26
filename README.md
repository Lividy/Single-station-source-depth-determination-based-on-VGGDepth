# Single-station-source-depth-determination-based-on-VGGDepth

This code includes data training, data testing, and visualization of prediction depth.

## 1. Environment Configuration
   
   conda env create -f env.yml (or env_all.yml if you need all my environment)

## 2. Download training data, testing data, and well-trained model folder

   #### Download 3 folder from my google drive:

   #### Copy three folder into demo_singlestation folder
      testing data folder: including one testing data of CESI station
      training data folder: including training_input and label, val_input, and val_label
      well_trained_model: CESI_epoch_300.pth is single station's well-trained model, NoED19_epoch_190.pth is generalized model
      
## 3. Network training 
      In demo_singlestation folder
         class_train.ipynb
      
## 4. Network testing
      In demo_singlestation folder
         class_test.ipynb

## 5. Visualization of prediction depth
      In demo_singlestation folder
      Plot_new_Depth_Location_single_station_DrawPicture.ipynb
      ## 
        ![Figure](https://github.com/Lividy/Single-station-source-depth-determination-based-on-VGGDepth/main/prediction.png)

         

