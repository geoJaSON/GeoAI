import arcgis
from arcgis.gis import GIS
from IPython.display import display
from arcgis.raster import ImageryLayer

gis_url = "https://ceswg.maps.arcgis.com"
username = "Jason.Jordan@SWD"
password = input("Password: ")
gis = GIS(gis_url,username,password)

gis_url = "https://swggis.com/portal"
username = "jason.jordan_swg"
password = "QAZw1s2x3"
gis2 = GIS(gis_url,username,password)

damaged_homes = gis.content.get('8a07510d660d4543ac3f5e990176e729') # flooded homes layer
damaged_homes


vexcel = gis.content.get("60b3a9f1e1d04470ac71c52d681083ff")
vexcel=vexcel.layers[0]

chips = export_training_data(vexcel, damaged_homes, "PNG", {"x":448,"y":448}, {"x":224,"y":224},
                             "PASCAL_VOC_rectangles", 75, "planetdemo")
