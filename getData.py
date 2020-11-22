import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import compute_class_weight

def getData():
  X = []
  Y = []
  void_label = -1

  train_path = r'C:\Users\Akhilash\Desktop\BackgroundSubtraction\unet\new'
  label_path = r'C:\Users\Akhilash\Desktop\BackgroundSubtraction\FgSegNet_v2-master\training_sets\CDnet2014_train\baseline\highway200'

  train_files = sorted(os.listdir(train_path))
  label_files = sorted(os.listdir(label_path))

  for i in range(len(train_files)):
    img = load_img(os.path.join(train_path, train_files[i]))
    img = img_to_array(img)
    X.append(img)

    img = load_img(os.path.join(label_path, label_files[i]), grayscale = True)
    img = img_to_array(img)
    shape = img.shape
    img /= 255.0
    img = img.reshape(-1)
    idx = np.where(np.logical_and(img > 0.25, img < 0.8))[0] # find non-ROI
    if len(idx) > 0:
      img[idx] = -1
    img = img.reshape(shape)
    img = np.floor(img)
    Y.append(img)

  X = np.asarray(X)
  Y = np.asarray(Y)

  idx = list(range(X.shape[0]))
  np.random.shuffle(idx)
  np.random.shuffle(idx)
  X = X[idx]
  Y = Y[idx]

  cls_weight_list = []
  for i in range(Y.shape[0]):
      y = Y[i].reshape(-1)
      idx = np.where(y!=void_label)[0]
      if(len(idx)>0):
          y = y[idx]
      lb = np.unique(y) #  0., 1
      cls_weight = compute_class_weight('balanced', lb , y)
      class_0 = cls_weight[0]
      class_1 = cls_weight[1] if len(lb)>1 else 1.0
          
      cls_weight_dict = {0:class_0, 1: class_1}
      cls_weight_list.append(cls_weight_dict)
          
  cls_weight_list = np.asarray(cls_weight_list)

  return [X, Y, cls_weight_list]
