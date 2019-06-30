import arcgis
from arcgis.gis import GIS
from IPython.display import display
from arcgis.raster import ImageryLayer

gis_url = "https://ceswg.maps.arcgis.com"
username = "Jason.Jordan@SWD"
password = "QAZw1s2x3"
gis = GIS(gis_url,username,password)

gis_url = "https://swggis.com/portal"
username = "jason.jordan_swg"
password = "QAZw1s2x3"
gis2 = GIS(gis_url,username,password)

flooded_homes = gis2.content.get('bc7eb7d4138f4b828101719d81db6c33') # flooded homes layer
flooded_homes


vexcel = gis.content.get("60b3a9f1e1d04470ac71c52d681083ff")
vexcel=vexcel.layers[0]
