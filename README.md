# Multi-Modal Fusion for Material Recognition in Autonomous Construction Machinery Operations

## Usage
This section outlines the recommended file structure and basic usage instructions for the project.

### RGB-D image acquisition
```plaintext
python d435i.py
```

#### Recommended File Structure
```plaintext
MMFMR/
│
├── data/                  
│   ├── examples/          
│   ├── splits/           
│   ├── rgbd-dataset/      # Dataset directory
│   └── weights/           # Trained model weights            
├── nets/                  # Moel configurations
│   ├── model.py           
│   └── ...           
├── utils/
│   ├── train_one_epoch.py           
│   └── ...
├── train.py              # Main script to train the model
├── requirements.txt      # Dependency file
└── README.md             # Project README
```

##### Data Preparation

Our self-constructed CMRD is available [in data.zip](https://huggingface.co/datasets/ysuck/CMRD/resolve/main/data.zip)

Washington RGB-D Object dataset is available [here](<https://rgbd-dataset.cs.washington.edu/dataset.html>).The `train.py` script utilizes a specific depth image processing method as detailed in the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314222000133). This method is crucial for preparing the dataset for effective training of our model. The following directory structure is a reference to run the `train.py` script

```plaintext
rgbd-dataset/
│
├── apple/                  
│   ├── apple_1/
│   │   ├── apple_1_1_1_crop.png
│   │   ├── apple_1_1_1_depthcrop.hdf5
│   │   ├── apple_1_1_1_depthcrop.png
│   │   ├── apple_1_1_1_loc.txt
│   │   └── ...           
│   └── ...
```
#### Running the Training Script
```bash
python train.py
```
