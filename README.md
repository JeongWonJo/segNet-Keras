# segNet-Keras
This is the edited version of “https://github.com/divamgupta/image-segmentation-keras”

## Prerequisites
python 2.7 
```
pip install keras==2.2.0 
pip install Tensorflow==1.8.0 
pip install opencv-python
pip install --upgrade theano
```
## Implemenation
### Download the pretrained VGG model
```
mkdir data
cd data
wget "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5"
```
### Download pretrained weights
Make a new folder ‘weights’ inside the ‘segNet-Keras’ folder. (it should be as follows: ~/segNet-Keras/weights/) <br>
Download pretrained weights from “https://drive.google.com/drive/folders/130h07bLqzJ9Cs9wu_QVjer0LLay0v-kD?usp=sharing” <br>
and unzip it inside the ‘weights‘ folder.

### Upload the image files
Make a new ‘test’ folder inside the ‘~/segNet-Keras/data/’ folder.<br>
Your directory should look like:<br>
```
segNet-Keras
  /data
    /test
```
Put images from which you want to get segmentation inside the ‘test’ folder.<br>

### Run the code
```
THEANO_FLAGS=device=gpu,floatX=float32  python  predict.py \
 --save_weights_path=weights/ex1 \
 --epoch_number=0 \
 --test_images="data/test/" \
 --output_path="data/predictions/" \
 --n_classes=10 \
 --model_name="vgg_segnet" 
```


<br><br>
## Train with your own dataset
Place images for training inside the ‘/segNet-Keras/data/train/images_prepped_train/’ folder and<br>
their annotations inside the ‘/segNet-Keras/data/train/annotations_prepped_train/’ folder.<br>
For your images for validating the trained model, place them inside the ‘/segNet-Keras/data/train/images_prepped_test/’ folder and<br>
their annotations inside the ‘/segNet-Keras/data/train/annotations_prepped_test/’ folder.<br>
<br><br>
Run the code below:
```
THEANO_FLAGS=device=gpu,floatX=float32  python  train.py \
 --save_weights_path=weights/ex1 \
 --train_images="data/train/images_prepped_train/" \
 --train_annotations="data/train/annotations_prepped_train/" \
 --val_images="data/train/images_prepped_test/" \
 --val_annotations="data/train/annotations_prepped_test/" \
 --n_classes=10 \
 --model_name="vgg_segnet" 
```
<br><br>
## Download the sample prepared dataset
Download and extract the following:<br>
https://drive.google.com/file/d/0B0d9ZiqAgFkiOHR1NTJhWVJMNEU/view?usp=sharing<br>
Place the folders in ‘~/segNet-Keras/data/train/’
