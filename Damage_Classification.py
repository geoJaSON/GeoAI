import arcpy
from arcpy.ia import *
arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("ImageAnalyst")

arcpy.ia.ExportTrainingDataForDeepLearning(in_raster=r"C:\Users\jason\Documents\NOAA Ida Imagery.lpkx",
    out_folder=r"C:\Users\jason\Documents\Training_256",
    in_class_data=r"C:\Users\jason\Documents\ArcGIS\Parcel_QA\Parcel_QA.gdb\noaa_buildings",
    image_chip_format="TIFF",
    tile_size_x=256,
    tile_size_y=256,
    stride_x=128,
    stride_y=128,
    output_nofeature_tiles="ONLY_TILES_WITH_FEATURES",
    metadata_format="RCNN_Masks",
    start_index=0,
    class_value_field="classvalue",
    buffer_radius=0,
    in_mask_polygons=None,
    rotation_angle=0,
    reference_system="MAP_SPACE",
    processing_mode="PROCESS_AS_MOSAICKED_IMAGE",
    blacken_around_feature="NO_BLACKEN",
    crop_mode="FIXED_SIZE")

# %%

from arcgis.learn import prepare_data, MaskRCNN
data = prepare_data(r'C:\Users\jason\Documents\Training\612', batch_size=1)
data.show_batch(rows=2)

#%%

model = MaskRCNN(data)
lr = model.lr_find()
lr

#%%

model.fit(epochs=10, lr=lr)
model.save('e20_256')
