# Single-station-source-depth-determination-based-on-VGGDepth

This code repository includes data training, data testing, and visualization for VGGDepth.

## 1. Environment Configuration

The neural networks are trained and implemented with Pytorch. You can set up a similar environment using the following commands.
- Install in the default environment:
   conda env create -f env.yml (or env_all.yml if you need all my environment)
   conda activate vggdepth
   
## 2. Download training data, testing data, and well-trained model folder

   Due to the large size of the data, we have placed it in the cloud storage.
   
   #### Download 3 folder from my google drive:
   
      https://drive.google.com/drive/folders/19SGqXPAGn1QaRUfflM3OUEruuCMgt4cZ?usp=sharing
      
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
 
![Figure](https://github.com/Lividy/Single-station-source-depth-determination-based-on-VGGDepth/raw/main/prediction.png)

         

