#!/usr/bin/env python
# coding: utf-8

# ## <span style="color:purple">ArcGIS API for Python: Real-time Person Detection</span>
# <img src="../img/webcam_detection.PNG" style="width: 100%"></img>
# 
# 
# ## Integrating ArcGIS with TensorFlow Deep Learning using the ArcGIS API for Python

# This notebook provides an example of integration between ArcGIS and deep learning frameworks like TensorFlow using the ArcGIS API for Python.
# 
# <img src="../img/ArcGIS_ML_Integration.png" style="width: 75%"></img>
# 
# We will leverage a model to detect objects on your device's video camera, and use these to update a feature service on a web GIS in real-time. As people are detected on your camera, the feature will be updated to reflect the detection. 

# ### Notebook Requirements: 

# #### 1. TensorFlow and Object Detection API

# This demonstration is designed to run using the TensorFlow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection)
# 
# Please follow the instructions found in that repository to install TensorFlow, clone the repository, and test a pre-existing model. 
# 
# Once you have followed those instructions, this notebook should be placed within the "object_detection" folder of that repository. Alternatively, you may leverage this notebook from another location but reference paths to the TensorFlow model paths and utilities will need to be adjusted. 

# #### 2. Access to ArcGIS Online or ArcGIS Enterprise 

# This notebook will make a connection to an ArcGIS Enterprise or ArcGIS Online organization to provide updates to a target feature service.
# 
# Please ensure you have access to an ArcGIS Enterprise or ArcGIS Online account with a feature service to serve as the target of your detection updates. The feature service should have a record with an boolean attribute (i.e. column with True or False possible options) named "Person_Found".

# # Import needed modules

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pandas as pd
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2


# We will use VideoCapture to connect to the device's web camera feed. The cv2 module helps here. 

# In[2]:


# Set our caption 
cap = cv2.VideoCapture(0)


# In[3]:


# This is needed since the notebook is meant to be run in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[4]:


from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[5]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model

# In[ ]:


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[8]:


category_index


# ## Helper code

# In[9]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# In[10]:


def object_counter(classes_arr, scores_arr, score_thresh=0.3):
    # Process the numpy array of classes from the model
    stacked_arr = np.stack((classes_arr, scores_arr), axis=-1)
    # Convert to pandas dataframe for easier querying
    detection_df = pd.DataFrame(stacked_arr)
    # Retrieve total count of cars with score threshold above param value
    detected_people =  detection_df[(detection_df[0] == 1.0) & (detection_df[1] > score_thresh)]

    
    people_count = len(detected_people)


    return people_count


# In[11]:


logging = "none"


# This is a helper function that takes the detection graph output tensor (np arrays), stacks the classes and scores, and determines if the class for a person (1) is available within a certain score and within a certain amount of objects

# In[12]:


def person_in_image(classes_arr, scores_arr, obj_thresh=5, score_thresh=0.5):
    stacked_arr = np.stack((classes_arr, scores_arr), axis=-1)
    person_found_flag = False
    for ix in range(obj_thresh):
        if 1.00000000e+00 in stacked_arr[ix]:
            if stacked_arr[ix][1] >= score_thresh:
                person_found_flag = True
            
    return person_found_flag


# # Establish Connection to GIS via ArcGIS API for Python

# ### Authenticate

# In[1]:


import arcgis
from arcgis.gis import GIS
from IPython.display import display


# In[14]:


gis_url = "https://swggis.com/portal"   # Replace with gis URL
username = "jason.jordan_swg"  # Replace with username
password = 'QAZw1s2x3'


# In[15]:


gis = GIS(gis_url, username, password)


# ### Retrieve the Object Detection Point Layer

# In[16]:


object_point = gis.content.search("GeoAI_TEST")[0]
object_point


# In[17]:


#object_layer = object_point.layers
#obj_fset = object_point.layers[0].query()
def search_layer(layer_name):
    search_results = gis.content.search(layer_name, item_type='Feature Layer')
    item = search_results[0]
    geofeatures = item.layers[0]
    return geofeatures

search_pnt = search_layer('GeoAI_TEST')
query_pnt = search_pnt.query(where='OBJECTID=1')


# In[18]:


object_features = query_pnt.features


# ### Test of Manual Update

# In[19]:


point_features = [f for f in object_features if f.attributes['objectid']==1][0]
point_features.attributes


# In[20]:


#features_for_update = []
point_features.attributes['person_found'] = "False"
point_features.attributes['person_count'] = 0
update_result = search_pnt.edit_features(updates=[point_features])


# # Detection

# In[ ]:


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=8, min_score_thresh=0.5)
            
            cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
                
            people_count = object_counter(np.squeeze(classes).astype(np.int32), np.squeeze(scores))

        
            if logging == "":
                print("/n")
                print("Detected {0} total objects...".format(str(total_count)))
                print("Detected {0} total vehicles...".format(str(vehicle_count)))
                print("Detected {0} cars...".format(str(car_count)))
                print("Detected {0} motorcycles...".format(str(motorcycle_count)))
                print("Detected {0} buses...".format(str(bus_count)))
                print("Detected {0} trucks...".format(str(truck_count)))
                print("Detected {0} pedestrians...".format(str(people_count)))
                print("Detected {0} bicycles...".format(str(bicycle_count)))
            
            elif logging == "simple":
                print("/n")
                print("Detected {0} total objects...".format(str(total_count)))
                print("Detected {0} total vehicles...".format(str(vehicle_count)))
                print("Detected {0} pedestrians...".format(str(people_count)))
                print("Detected {0} bicycles...".format(str(bicycle_count)))
                
            elif logging == "cars":
                print("/n")
                print("Detected {0} cars...".format(str(car_count)))
                
            elif logging == "none":
                print("Detected {0} pedestrians...".format(str(people_count)))

                
            person_found = person_in_image(np.squeeze(classes).astype(np.int32), 
                                           np.squeeze(scores), 
                                           obj_thresh=2)
            
            features_for_update = []
            point_features.attributes['person_found'] = str(person_found)
            point_features.attributes['person_count'] = str(people_count)
            features_for_update.append(point_features)
            search_pnt.edit_features(updates=features_for_update)


# In[ ]:





# In[ ]:




