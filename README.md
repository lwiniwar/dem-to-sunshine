# dem-to-sunshine

## Installation
Install the required packages:
```shell
pip install rasterio numpy matplotlib pandas astropy pyproj tyro scipy
```

Then clone and `cd`:
```
git clone https://github.com/lwiniwar/dem-to-sunshine.git
cd dem-to-sunshine
```

## Download raster files
e.g. from
https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dom20 (for Bavaria)
or
https://tiris.maps.arcgis.com/apps/webappviewer/index.html?id=5e3071044cb44e76843d110baef8b138 (for Tyrol)

## Locate the POI
I do this by opening the DEM in QGIS and right-clicking on the location -> "Copy coordinates" -> "WGS84"

## Running the tool
run using

```shell
main.py --args.dem-path demo/32693_5333_20_DOM.tif --args.lat 48.12466249 --args.lon 11.59677922 --args.height-above-ground 7.3 --args.output-prefix muc
```