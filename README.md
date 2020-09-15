# Kaggle Carvana Image Masking Challenge solution with Keras
This solution was based on [Heng CherKeng's code for PyTorch](https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37208). I kindly thank him for sharing his work. 128x128, 256x256, 512x512 and 1024x1024 U-nets are implemented. Public LB scores for each U-net are:

| U-net | LB score |
| ----- | -------- |
| 128x128 | 0.990 |
| 256x256 | 0.994 |

---

## Updates
* Added loss with weighted boundary (thanks to [lyakaap](https://www.kaggle.com/lyakaap/weighing-boundary-pixels-loss-script-by-keras2))
* Added Hue/Saturation/Value augmentation.
* Switched to RMSprop optimizer as default.
* Using *Binary Crossentropy Dice Loss* in place of *Binary Crossentropy*
* Callbacks now use *val_dice_loss* as a metric in place of *val_loss*

---

## Requirements
* Keras 2.0 w/ TF backend
* sklearn
* cv2
* tqdm
* h5py

---

## Usage

### Data
Place '*train*', '*train_masks*' and '*test*' data folders in the '*input*' folder.

Convert training masks to *.png* format. You can do this with: 

`python formatTransformer.py` 

in the '*train_masks*' data folder.

### Train
Run `carvana_solution.ipynb` to train the model.

### Test and submit
Run `carvana_solution.ipynb` to make predictions on test data and generate submission.
