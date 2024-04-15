# multiscale_net
This is the code implementation of "An efficient deep neural network for automatic classification of acute intracranial hemorrhages in brain CT scans"


## Data preprocessing
   Download ICH datasets from [RSNA website](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) and type:

   ```bash
   python preprocessing.py --dcm-dir address_of_ICH_datasets --img-dr address_of_save_directories
   ```

## Requirements
   ```bash
   pip install numpy pillow cv2 torch torchvision sklearn pandas
   ```

## Results
   Run the following command:
    
   ```bash
   python train.py --img-dr address_of_save_directories
   ```



