import pandas as pd
from arcgis.features import GeoAccessor, GeoSeriesAccessor
sdf = pd.DataFrame.spatial.from_featureclass(r'E:\vol001\aifinds.shp')

sdf1 = sdf.loc[sdf.groupby("GEO_ID")["Shape_Area"].idxmax()]

sdf1['coordinates'] = sdf1.SHAPE.geom.centroid




sdf['']
sdf=sdf.groupby('GEO_ID').max('Shape_Area')
sdf1['Lat'] = sdf1['coordinates'].str[0]
sdf1['long'] = sdf1['coordinates'].str[1]
sdf1.spatial.to_featureclass(location=r'E:\vol001\aifinds1.shp')
sdf1.to_csv(r'E:\vol001\aifinds.csv')



#Process Lidar/Imagery
#Building Footprint Extraction
#Join parcels to footprints (has their center in)
#Query out largest area footprint per parcel
#Get centroid of remainder footprints
#Extract other attributes (sqft, height, roof form, etc)
