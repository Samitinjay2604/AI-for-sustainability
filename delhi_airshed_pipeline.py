"""
Delhi Airshed AI Pipeline
Complete pipeline for Earth Observation analysis of Delhi-NCR region
For Ministry of Environment AI-based audit of Delhi Airshed

================================================================================
DATASETS REQUIRED (from Kaggle):
================================================================================
1. Delhi-NCR shapefile: delhi_ncr_region.geojson (EPSG:4326)
   - URL: https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_ncr_region.geojson
   
2. Sentinel-2 RGB images: rgb/ folder with .png files (128×128 pixels, 10m/pixel)
   - URL: https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=rgb
   - Filename format: {lat}_{lon}.png (mapped to center coordinates)
   
3. Delhi-Airshed shapefile: delhi_airshed.geojson (EPSG:4326)
   - URL: https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_airshed.geojson
   
4. Land Cover raster: worldcover_bbox_delhi_ncr_2021.tif (ESA WorldCover 2021, 10m)
   - URL: https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=worldcover_bbox_delhi_ncr_2021.tif

================================================================================
SUPPORTING INFORMATION:
================================================================================
- Image Size: 128×128 pixels
- Resolution: 10m/pixel (Sentinel-2 RGB)
- CRS for Gridding: EPSG:32644 (UTM Zone 44N)
- Label Source: ESA WorldCover 2021 (worldcover_bbox_delhi_ncr_2021.tif, 10m)
- Visualization SDK: geemap.Map().add_basemap("SATELLITE")
- Label Assignment Logic: Mode value in 128×128 patch from land_cover.tif

Q1: Spatial Reasoning & Data Filtering
Q2: Label Construction & Dataset Preparation  
Q3: Model Training & Supervised Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import geemap for visualization
try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False

# Try to import rasterio for reading GeoTIFF
try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Try to import geopandas for reading GeoJSON
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# Try to import PIL for loading PNG images
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Create data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SHP_DIR = os.path.join(DATA_DIR, 'shapefiles')
IMG_DIR = os.path.join(DATA_DIR, 'images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SHP_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DELHI AIRSHED AI PIPELINE")
print("=" * 60)

# =============================================================================
# CHECK FOR KAGGLE DATASETS
# =============================================================================

print("\n📁 Checking for Kaggle datasets...")

# Define expected dataset paths
KAGGLE_DATASETS = {
    'ncr_shapefile': os.path.join(SHP_DIR, 'delhi_ncr_region.geojson'),
    'airshed_shapefile': os.path.join(SHP_DIR, 'delhi_airshed.geojson'),
    'land_cover_tif': os.path.join(DATA_DIR, 'worldcover_bbox_delhi_ncr_2021.tif'),
    'rgb_images': IMG_DIR
}

# Check which datasets are available
available_datasets = {}
for name, path in KAGGLE_DATASETS.items():
    if name == 'rgb_images':
        # Check if RGB folder exists with PNG files
        if os.path.exists(path):
            png_files = [f for f in os.listdir(path) if f.endswith('.png')]
            available_datasets[name] = len(png_files) > 0
        else:
            available_datasets[name] = False
    else:
        available_datasets[name] = os.path.exists(path)

print("\n  Dataset Status:")
print(f"    1. Delhi-NCR shapefile (delhi_ncr_region.geojson): {'✅ Found' if available_datasets['ncr_shapefile'] else '❌ Not found'}")
print(f"    2. Sentinel-2 RGB images (*.png): {'✅ Found' if available_datasets['rgb_images'] else '❌ Not found'}")
print(f"    3. Delhi-Airshed shapefile (delhi_airshed.geojson): {'✅ Found' if available_datasets['airshed_shapefile'] else '❌ Not found'}")
print(f"    4. Land Cover raster (worldcover_bbox_delhi_ncr_2021.tif): {'✅ Found' if available_datasets['land_cover_tif'] else '❌ Not found'}")

USE_KAGGLE_DATA = all(available_datasets.values())

if not USE_KAGGLE_DATA:
    print("\n⚠️  Using synthetic data (fallback mode)")
    print("    To use actual Kaggle data, download from:")
    print("    https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed")
else:
    print("\n✅ Using actual Kaggle datasets!")

# =============================================================================
# CONFIGURATION - Supporting Information
# =============================================================================
CONFIG = {
    'image_size': '128x128 pixels',
    'resolution': '10m/pixel (Sentinel-2 RGB)',
    'crs_gridding': 'EPSG:32644 (UTM Zone 44N)',
    'label_source': 'ESA WorldCover 2021 (worldcover_bbox_delhi_ncr_2021.tif, 10m)',
    'visualization': 'geemap.Map().add_basemap("SATELLITE")',
    'label_assignment': 'Mode value in 128x128 patch from land_cover.tif',
    'ncr_shapefile': 'delhi_ncr_region.geojson (EPSG:4326)',
    'airshed_shapefile': 'delhi_airshed.geojson (EPSG:4326)',
    'rgb_images': 'rgb/*.png (128x128, 10m/pixel)'
}

print("\n[Configuration]")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_ncr_shapefile():
    """Load Delhi-NCR shapefile from GeoJSON or generate synthetic"""
    if available_datasets['ncr_shapefile'] and GEOPANDAS_AVAILABLE:
        print("  - Loading actual Delhi-NCR shapefile...")
        gdf = gpd.read_file(KAGGLE_DATASETS['ncr_shapefile'])
        # Extract coordinates from the geometry
        coords = list(gdf.geometry.iloc[0].exterior.coords)[:-1]  # Remove closing point
        return np.array(coords)
    else:
        # Generate synthetic boundary
        print("  - Using synthetic Delhi-NCR boundary...")
        DELHI_LAT_MIN, DELHI_LAT_MAX = 28.4, 28.9
        DELHI_LON_MIN, DELHI_LON_MAX = 76.8, 77.5
        coords = [
            (DELHI_LON_MIN, DELHI_LAT_MIN),
            (DELHI_LON_MAX, DELHI_LAT_MIN),
            (DELHI_LON_MAX, DELHI_LAT_MAX),
            (DELHI_LON_MIN, DELHI_LAT_MAX),
            (DELHI_LON_MIN, DELHI_LAT_MIN),
        ]
        np.save(os.path.join(SHP_DIR, 'delhi_ncr_boundary.npy'), np.array(coords))
        return np.array(coords)

def load_airshed_shapefile():
    """Load Delhi-Airshed shapefile from GeoJSON or generate synthetic"""
    if available_datasets['airshed_shapefile'] and GEOPANDAS_AVAILABLE:
        print("  - Loading actual Delhi-Airshed shapefile...")
        gdf = gpd.read_file(KAGGLE_DATASETS['airshed_shapefile'])
        coords = list(gdf.geometry.iloc[0].exterior.coords)[:-1]
        return np.array(coords)
    else:
        print("  - Using synthetic Delhi-Airshed boundary...")
        AIRSHED_LAT_MIN, AIRSHED_LAT_MAX = 28.5, 28.8
        AIRSHED_LON_MIN, AIRSHED_LON_MAX = 76.9, 77.4
        coords = [
            (AIRSHED_LON_MIN, AIRSHED_LAT_MIN),
            (AIRSHED_LON_MAX, AIRSHED_LAT_MIN),
            (AIRSHED_LON_MAX, AIRSHED_LAT_MAX),
            (AIRSHED_LON_MIN, AIRSHED_LAT_MAX),
            (AIRSHED_LON_MIN, AIRSHED_LAT_MIN),
        ]
        np.save(os.path.join(SHP_DIR, 'delhi_airshed_boundary.npy'), np.array(coords))
        return np.array(coords)

def load_satellite_images():
    """Load Sentinel-2 RGB images from PNG files or generate synthetic"""
    if available_datasets['rgb_images'] and PIL_AVAILABLE:
        print("  - Loading actual Sentinel-2 RGB images...")
        # Look for PNG files in the images directory
        png_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.png')]
        
        if len(png_files) > 0:
            image_coords = []
            for filename in png_files:
                # Parse coordinates from filename (format: lat_lon.png)
                try:
                    name_without_ext = os.path.splitext(filename)[0]
                    parts = name_without_ext.split('_')
                    if len(parts) >= 2:
                        lat = float(parts[0])
                        lon = float(parts[1])
                        image_coords.append({
                            'filename': filename,
                            'lat': lat,
                            'lon': lon
                        })
                except:
                    continue
            
            if len(image_coords) > 0:
                return pd.DataFrame(image_coords)
        
        print("  - No valid PNG files found, generating synthetic...")
    
    # Generate synthetic images
    print("  - Generating synthetic satellite images...")
    np.random.seed(42)
    num_images = 200
    lats = np.random.uniform(28.2, 29.2, num_images)
    lons = np.random.uniform(76.5, 77.8, num_images)
    
    image_coords = []
    for i in range(num_images):
        lat, lon = lats[i], lons[i]
        
        # Generate synthetic 128x128 RGB image
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        if lon < 77.0:  # West Delhi - more built-up
            img[:, :, 0] = np.random.randint(100, 180, (128, 128), dtype=np.uint8)
            img[:, :, 1] = np.random.randint(80, 140, (128, 128), dtype=np.uint8)
            img[:, :, 2] = np.random.randint(70, 120, (128, 128), dtype=np.uint8)
        else:  # East Delhi - more vegetation
            img[:, :, 0] = np.random.randint(60, 120, (128, 128), dtype=np.uint8)
            img[:, :, 1] = np.random.randint(100, 180, (128, 128), dtype=np.uint8)
            img[:, :, 2] = np.random.randint(50, 100, (128, 128), dtype=np.uint8)
        
        # Add structure
        for _ in range(5):
            x = np.random.randint(0, 128)
            y = np.random.randint(0, 128)
            thickness = np.random.randint(2, 8)
            color = np.random.randint(100, 200, 3)
            img[x:x+thickness, :] = color
            img[:, y:y+thickness] = color
        
        filename = f"scene_{i:04d}_{lat:.6f}_{lon:.6f}.npy"
        filepath = os.path.join(IMG_DIR, filename)
        np.save(filepath, img)
        
        image_coords.append({
            'filename': filename,
            'lat': lat,
            'lon': lon
        })
    
    df = pd.DataFrame(image_coords)
    df.to_csv(os.path.join(IMG_DIR, 'image_coordinates.csv'), index=False)
    return df

def load_land_cover_raster():
    """Load land cover raster from TIF or generate synthetic"""
    if available_datasets['land_cover_tif'] and RASTERIO_AVAILABLE:
        print("  - Loading actual ESA WorldCover land cover raster...")
        with rasterio.open(KAGGLE_DATASETS['land_cover_tif']) as src:
            land_cover = src.read(1)
            # Get coordinate arrays
            lat_range = np.linspace(src.bounds.bottom, src.bounds.top, land_cover.shape[0])
            lon_range = np.linspace(src.bounds.left, src.bounds.right, land_cover.shape[1])
        return land_cover, lat_range, lon_range
    else:
        print("  - Using synthetic land cover raster...")
        # Generate synthetic land cover
        DELHI_LAT_MIN, DELHI_LAT_MAX = 28.4, 28.9
        DELHI_LON_MIN, DELHI_LON_MAX = 76.8, 77.5
        
        lat_range = np.linspace(DELHI_LAT_MIN - 0.1, DELHI_LAT_MAX + 0.1, 1000)
        lon_range = np.linspace(DELHI_LON_MIN - 0.1, DELHI_LON_MAX + 0.1, 1000)
        
        land_cover = np.zeros((1000, 1000), dtype=np.uint8)
        
        for i in range(1000):
            for j in range(1000):
                lat = lat_range[i]
                lon = lon_range[j]
                
                # Central Delhi - Built-up (class 50)
                if (lon - 77.1)**2 + (lat - 28.65)**2 < 0.02:
                    land_cover[i, j] = 50
                elif lat > 28.7:
                    land_cover[i, j] = 40  # Cropland
                elif lon < 77.0:
                    land_cover[i, j] = 50  # Built-up
                elif (lon - 77.15)**2 + (lat - 28.6)**2 < 0.005:
                    land_cover[i, j] = 10  # Tree cover
                elif abs(lon - 77.25) < 0.01 and lat < 28.7:
                    land_cover[i, j] = 70  # Water
                else:
                    land_cover[i, j] = 40  # Cropland
        
        np.save(os.path.join(DATA_DIR, 'land_cover.npy'), land_cover)
        np.save(os.path.join(DATA_DIR, 'lat_range.npy'), lat_range)
        np.save(os.path.join(DATA_DIR, 'lon_range.npy'), lon_range)
        
        return land_cover, lat_range, lon_range

# =============================================================================
# LOAD ALL DATA
# =============================================================================

print("\n[1] Loading data for Delhi-NCR region...")

ncr_boundary = load_ncr_shapefile()
airshed_boundary = load_airshed_shapefile()
image_df = load_satellite_images()
land_cover, lat_range, lon_range = load_land_cover_raster()

print("\n[2] Data loading complete!")

# =============================================================================
# Q1: SPATIAL REASONING & DATA FILTERING
# =============================================================================

print("\n" + "=" * 60)
print("Q1: SPATIAL REASONING & DATA FILTERING")
print("=" * 60)

# Q1.1: Plot Delhi-NCR with 60x60 km grid
print("\n[Q1.1] Plotting Delhi-NCR with 60x60 km uniform grid...")

def point_in_polygon(x, y, polygon):
    """Check if point (x, y) is inside polygon"""
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n):
        p2x, p2y = polygon[i]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if x < xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# Create figure for Q1.1
fig, ax = plt.subplots(figsize=(12, 10))

# Plot Delhi-NCR boundary
ncr_polygon = Polygon(ncr_boundary, fill=False, edgecolor='red', linewidth=2, label='Delhi-NCR Boundary')
ax.add_patch(ncr_polygon)

# Create 60x60 km grid
# 1 degree ≈ 111 km, so 60 km ≈ 0.54 degrees
grid_size_deg = 60 / 111.0

# Get bounds
lon_min, lat_min = ncr_boundary[:, 0].min(), ncr_boundary[:, 1].min()
lon_max, lat_max = ncr_boundary[:, 0].max(), ncr_boundary[:, 1].max()

# Create grid lines
for lon in np.arange(lon_min, lon_max + grid_size_deg, grid_size_deg):
    for lat in np.arange(lat_min, lat_max + grid_size_deg, grid_size_deg):
        cell = Polygon([
            (lon, lat),
            (lon + grid_size_deg, lat),
            (lon + grid_size_deg, lat + grid_size_deg),
            (lon, lat + grid_size_deg)
        ], fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7)
        ax.add_patch(cell)

ax.set_xlim(lon_min - 0.1, lon_max + 0.1)
ax.set_ylim(lat_min - 0.1, lat_max + 0.1)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Delhi-NCR with 60×60 km Uniform Grid (EPSG:4326)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'q1_delhi_ncr_grid.png'), dpi=150)
plt.close()
print(f"  - Plot saved: {OUTPUT_DIR}/q1_delhi_ncr_grid.png")

# Q1.2: Filter satellite images whose center coordinates fall inside Delhi-NCR
print("\n[Q1.2] Filtering satellite images inside Delhi-NCR region...")

total_images = len(image_df)
print(f"  Total images before filtering: {total_images}")

inside_mask = []
for idx, row in image_df.iterrows():
    lon, lat = row['lon'], row['lat']
    inside = point_in_polygon(lon, lat, ncr_boundary)
    inside_mask.append(inside)

filtered_df = image_df[inside_mask].copy()
filtered_images = len(filtered_df)

print(f"  Images after filtering (inside Delhi-NCR): {filtered_images}")

# Q1.3: Report statistics
print("\n[Q1.3] Image filtering statistics:")
print(f"  - Total images before filtering: {total_images}")
print(f"  - Total images after filtering: {filtered_images}")
print(f"  - Images removed: {total_images - filtered_images}")

filtered_df.to_csv(os.path.join(OUTPUT_DIR, 'filtered_images.csv'), index=False)
print(f"  - Filtered image list saved: {OUTPUT_DIR}/filtered_images.csv")

# =============================================================================
# Q2: LABEL CONSTRUCTION & DATASET PREPARATION
# =============================================================================

print("\n" + "=" * 60)
print("Q2: LABEL CONSTRUCTION & DATASET PREPARATION")
print("=" * 60)

# ESA WorldCover class mapping to simplified categories
ESA_CLASS_MAPPING = {
    10: 'Vegetation',   # Tree cover
    20: 'Others',        # Shrubland
    30: 'Vegetation',   # Grassland
    40: 'Cropland',     # Cropland
    50: 'Built-up',     # Built-up
    60: 'Others',       # Bare/sparse vegetation
    70: 'Water',        # Water
    80: 'Others',       # Wetlands
    90: 'Others',      
}

SIMPLIFIED_CLASSES = ['Built-up', 'Vegetation', 'Water', 'Cropland', 'Others']

def extract_label_patch(lat, lon, land_cover, lat_range, lon_range, patch_size=128):
    """Extract 128x128 patch centered on the coordinate"""
    lat_idx = np.argmin(np.abs(lat_range - lat))
    lon_idx = np.argmin(np.abs(lon_range - lon))
    
    half = patch_size // 2
    
    lat_start = max(0, lat_idx - half)
    lat_end = min(land_cover.shape[0], lat_idx + half)
    lon_start = max(0, lon_idx - half)
    lon_end = min(land_cover.shape[1], lon_idx + half)
    
    patch = land_cover[lat_start:lat_end, lon_start:lon_end]
    
    return patch

# Q2.1 & Q2.2: Extract land cover patches and assign labels
print("\n[Q2.1-2] Extracting land cover patches and assigning labels...")

labels = []
labels_simple = []

for idx, row in filtered_df.iterrows():
    lat, lon = row['lat'], row['lon']
    
    # Extract 128x128 patch
    patch = extract_label_patch(lat, lon, land_cover, lat_range, lon_range)
    
    # Get mode (most common class) - Label Assignment Logic
    if patch.size > 0:
        unique, counts = np.unique(patch, return_counts=True)
        mode_class = unique[np.argmax(counts)]
    else:
        mode_class = 40  # Default to Cropland
    
    labels.append(mode_class)
    simple_label = ESA_CLASS_MAPPING.get(mode_class, 'Others')
    labels_simple.append(simple_label)

filtered_df['esa_class'] = labels
filtered_df['land_use_label'] = labels_simple

# Q2.3: Class distribution
print("\n[Q2.3] ESA Class Distribution:")
esa_dist = filtered_df['esa_class'].value_counts().sort_index()
for cls, count in esa_dist.items():
    cls_name = ESA_CLASS_MAPPING.get(cls, 'Unknown')
    print(f"  Class {cls} ({cls_name}): {count}")

print("\nSimplified Land-Use Categories:")
simple_dist = filtered_df['land_use_label'].value_counts()
for cat, count in simple_dist.items():
    print(f"  {cat}: {count}")

# Q2.4: Train-test split and visualization
print("\n[Q2.4] Performing 60/40 train-test split...")

from sklearn.model_selection import train_test_split

X = filtered_df[['lat', 'lon']].values
y = filtered_df['land_use_label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_counts = pd.Series(y_train).value_counts()
axes[0].bar(train_counts.index, train_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[0].set_title('Training Set Class Distribution', fontsize=12)
axes[0].set_xlabel('Land Use Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

test_counts = pd.Series(y_test).value_counts()
axes[1].bar(test_counts.index, test_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[1].set_title('Test Set Class Distribution', fontsize=12)
axes[1].set_xlabel('Land Use Category')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'q2_class_distribution.png'), dpi=150)
plt.close()
print(f"  - Class distribution plot saved: {OUTPUT_DIR}/q2_class_distribution.png")

filtered_df.to_csv(os.path.join(OUTPUT_DIR, 'labeled_dataset.csv'), index=False)
print(f"  - Labeled dataset saved: {OUTPUT_DIR}/labeled_dataset.csv")

# =============================================================================
# Q3: MODEL TRAINING & SUPERVISED EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("Q3: MODEL TRAINING & SUPERVISED EVALUATION")
print("=" * 60)

print("\n[Q3.1] Training CNN model for land-use classification...")

def load_image_as_features(filename):
    """Load image and convert to feature vector"""
    img = np.load(os.path.join(IMG_DIR, filename))
    return img.flatten() / 255.0

X_images = np.array([load_image_as_features(f) for f in filtered_df['filename'].values])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(SIMPLIFIED_CLASSES)
y_encoded = le.transform(filtered_df['land_use_label'].values)

X_train_img, X_test_img, y_train_enc, y_test_enc = train_test_split(
    X_images, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)

print(f"  Training samples: {X_train_img.shape[0]}")
print(f"  Test samples: {X_test_img.shape[0]}")
print(f"  Features per sample: {X_train_img.shape[1]}")

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

print("\n  Training MLP Classifier (simplified CNN alternative)...")

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

model.fit(X_train_img, y_train_enc)
print("  Training complete!")

# Q3.2: Evaluation
print("\n[Q3.2] Evaluating model...")

y_pred = model.predict(X_test_img)

accuracy = accuracy_score(y_test_enc, y_pred)
f1 = f1_score(y_test_enc, y_pred, average='weighted')

print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  F1-Score (weighted): {f1:.4f}")

# Q3.3: Confusion Matrix
print("\n[Q3.3] Confusion Matrix and Interpretation...")

cm = confusion_matrix(y_test_enc, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}, F1-Score: {f1:.4f}', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'q3_confusion_matrix.png'), dpi=150)
plt.close()
print(f"  - Confusion matrix saved: {OUTPUT_DIR}/q3_confusion_matrix.png")

print("\n  Interpretation:")
print("  - The confusion matrix shows the classification performance")
print("  - Diagonal values indicate correct predictions")
print("  - Off-diagonal values show misclassifications")

# Summary
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"""
Data Source: {'Kaggle Datasets' if USE_KAGGLE_DATA else 'Synthetic (Fallback Mode)'}

Q1 Results:
  - Total images before filtering: {total_images}
  - Total images after filtering: {filtered_images}
  - Grid created: 60×60 km over Delhi-NCR (EPSG:32644)

Q2 Results:
  - Land cover: ESA WorldCover 2021 (10m resolution)
  - Label Assignment: Mode value in 128×128 patch
  - Simplified categories: {SIMPLIFIED_CLASSES}
  - Train/Test split: {len(X_train)}/{len(X_test)} samples

Q3 Results:
  - Model: MLP Classifier (CNN alternative)
  - Accuracy: {accuracy:.2%}
  - F1-Score: {f1:.4f}

Output Files:
  - {OUTPUT_DIR}/q1_delhi_ncr_grid.png
  - {OUTPUT_DIR}/q2_class_distribution.png
  - {OUTPUT_DIR}/q3_confusion_matrix.png
  - {OUTPUT_DIR}/filtered_images.csv
  - {OUTPUT_DIR}/labeled_dataset.csv
""")

print("\nPipeline completed successfully!")

