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

damaged_homes = gis.content.get('b4e498c037fc4cc5b23bc1121a60e42a') # flooded homes layer
damaged_homes

destroyed_homes = gis.content.get('66cf5f24d1024f51a722488770d0079e')
destroyed_homes


vexcel = gis.content.get("60b3a9f1e1d04470ac71c52d681083ff")
vexcel=vexcel.layers[0]
