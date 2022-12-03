import pandas as pd
import os
import shutil
from pathlib import Path

from arcgis.learn import prepare_data, MaskRCNN
data = prepare_data(r'C:\Users\jason\Documents\Training\612', batch_size=1)
data.show_batch(rows=2)

#%%



model_def_path = r"C:\Users\jason\Documents\Training\612\models\e20_256\e20_256.dlpk" # Esri Windows and Doors.emd file is in the extracted contents of the DLPK

fcnn = MaskRCNN.from_model(model_def_path, data=data)


fcnn.show_results(thresh=0.9)

model = MaskRCNN(data)
lr = model.lr_find()
lr

#%%

model.fit(epochs=10, lr=lr)
model.save('e12_256')
