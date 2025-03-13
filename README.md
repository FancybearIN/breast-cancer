# Breast-Cancer-Detection-Using-Deep-Learning

### Implemented by CNN using Keras and TensorFlow and published in Informatics in Medicine Unlocked Volume 16, 2019, 100231 "Cancer diagnosis in histopathological image: CNN based approach" https://www.sciencedirect.com/science/article/pii/S2352914819301133

## Requirements:
command: `pip3 install -r requirements.txt`

### General Requirements: 
1. Windows or Linux PC with at least 2 GB of RAM.
2. Python 3

### Python Requirements: 
1. numpy==1.18.1
2. pydot==1.4.1
3. graphviz==0.13.2
4. h5py==2.10.0
5. matplotlib==3.1.3
6. keras==2.1.1
7. tensorflow==1.14.0
8. cv2==4.2.0
9. sklearn==0.22.1

## Setup:
1. Clone this repository.
2. Install dependencies listed in `requirements.txt`.
3. Download the dataset provided by https://ieeexplore.ieee.org/document/7312934.
4. Put all benign and malignant images in folders named "benign" and "malignant" respectively, and move both of them into the `data` folder in the repository home.
5. Using the `tree` command, make sure the directory structure is the same as `directory_structure.txt`.
6. Run `python3 main.py` to reproduce results.
7. Run `python3 guifinal.py` to reproduce the same results using the GUI.
8. Run `python3 prediction.py` for predictions done on the trained model.



