// ===============================
// 1. LOAD ASSETS
// ===============================
var nuremberg = ee.FeatureCollection("projects/ee-hpraveenhegde359/assets/nuremberg_boundary");

// ===============================
// 2. LOAD S2 (2020) & WORLDCOVER (2021)
// ===============================
var s2Collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterBounds(nuremberg)
  .filterDate('2020-01-01', '2020-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20));

var s2 = s2Collection.median().clip(nuremberg).select(['B2','B3','B4','B5','B6','B7','B8','B11','B12']);
var imageCount = s2Collection.size();

var lc = ee.Image("ESA/WorldCover/v200/2021").select('Map');

// One-hot class bands
var lcBands = ee.Image([
  lc.eq(10).rename('tree_cover'),
  lc.eq(30).rename('grassland'),
  lc.eq(40).rename('cropland'),
  lc.eq(50).rename('built_up'),
  lc.eq(60).rename('bare_sparse_vegetation'),
  lc.eq(80).rename('water')
]);

// Dominant class & Fractions at 20m
var classLabel20m = lc
  .reduceResolution({reducer: ee.Reducer.mode(), maxPixels: 1024})
  .reproject({crs: 'EPSG:32632', scale: 20})
  .rename('class_label');

var lcFractions20m = lcBands
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject({crs: 'EPSG:32632', scale: 20});

var coordsImg = ee.Image.pixelCoordinates('EPSG:32632');

var combinedImage = s2.addBands(lcFractions20m).addBands(classLabel20m).addBands(coordsImg);

// ===============================
// 3. SAMPLE PIXELS (Optimized)
// ===============================
var pixelData = combinedImage.sample({
  region: nuremberg.geometry(),
  scale: 20,
  projection: 'EPSG:32632',
  geometries: true, // This generates the .geo column automatically
  dropNulls: true
});

// Add metadata (year and count)
var cleanData = pixelData.map(function(f) {
  return f.set({
    'year': 2020,
    'image_count': imageCount
  });
});

// ===============================
// 4. EXPORT
// ===============================
Export.table.toDrive({
  collection: cleanData,
  description: 'df_2020_S2_2021_Labels_with_Geo',
  fileFormat: 'CSV',
  selectors: [
    'system:index', 
    'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8',
    'bare_sparse_vegetation', 'built_up', 'cropland',
    'grassland', 'image_count', 'tree_cover', 'water',
    'x', 'y', 'year', 'class_label', '.geo' // .geo is now included
  ]
});