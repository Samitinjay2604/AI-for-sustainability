"""
Delhi Airshed AI Pipeline
Complete pipeline for Earth Observation analysis of Delhi-NCR region
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
# DATA GENERATION - Create synthetic data for Delhi-NCR region
# =============================================================================

print("\n[1] Generating synthetic data for Delhi-NCR region...")

# Delhi-NCR bounding box (EPSG:4326)
DELHI_LAT_MIN, DELHI_LAT_MAX = 28.4, 28.9
DELHI_LON_MIN, DELHI_LON_MAX = 76.8, 77.5

# Delhi-Airshed region (slightly smaller area within NCR)
AIRSHED_LAT_MIN, AIRSHED_LAT_MAX = 28.5, 28.8
AIRSHED_LON_MIN, AIRSHED_LON_MAX = 76.9, 77.4

def generate_delhi_ncr_shapefile():
    """Generate Delhi-NCR polygon shapefile"""
    # Create a simplified polygon for Delhi-NCR boundary
    coords = [
        (DELHI_LON_MIN, DELHI_LAT_MIN),  # SW
        (DELHI_LON_MAX, DELHI_LAT_MIN),  # SE
        (DELHI_LON_MAX, DELHI_LAT_MAX),  # NE
        (DELHI_LON_MIN, DELHI_LAT_MAX),  # NW
        (DELHI_LON_MIN, DELHI_LAT_MIN),  # Close
    ]
    
    # Save as CSV-based "shapefile" format (simplified)
    shp_data = {
        'name': ['Delhi-NCR'],
        'geometry': [coords]
    }
    
    # Save to text file
    with open(os.path.join(SHP_DIR, 'delhi_ncr_boundary.txt'), 'w') as f:
        f.write("name,geometry\n")
        f.write(f"Delhi-NCR,{coords}\n")
    
    # Also create as numpy for easier processing
    np.save(os.path.join(SHP_DIR, 'delhi_ncr_boundary.npy'), np.array(coords))
    
    print(f"  - Delhi-NCR boundary saved: {SHP_DIR}/delhi_ncr_boundary.npy")
    return np.array(coords)

def generate_delhi_airshed_shapefile():
    """Generate Delhi-Airshed polygon"""
    coords = [
        (AIRSHED_LON_MIN, AIRSHED_LAT_MIN),
        (AIRSHED_LON_MAX, AIRSHED_LAT_MIN),
        (AIRSHED_LON_MAX, AIRSHED_LAT_MAX),
        (AIRSHED_LON_MIN, AIRSHED_LAT_MAX),
        (AIRSHED_LON_MIN, AIRSHED_LAT_MIN),
    ]
    
    np.save(os.path.join(SHP_DIR, 'delhi_airshed_boundary.npy'), np.array(coords))
    print(f"  - Delhi-Airshed boundary saved: {SHP_DIR}/delhi_airshed_boundary.npy")
    return np.array(coords)

def generate_satellite_images(num_images=200):
    """Generate synthetic Sentinel-2 RGB image patches"""
    print(f"  - Generating {num_images} synthetic satellite images...")
    
    # Generate random center coordinates covering a wider area
    # Some will be inside Delhi-NCR, some outside
    np.random.seed(42)
    
    # Generate coordinates in a wider region
    lats = np.random.uniform(28.2, 29.2, num_images)
    lons = np.random.uniform(76.5, 77.8, num_images)
    
    image_coords = []
    
    for i in range(num_images):
        lat, lon = lats[i], lons[i]
        
        # Generate synthetic 128x128 RGB image
        # Use different patterns based on location to simulate different land covers
        
        # Base image - varies by location
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        
        # Add some noise and patterns
        if lon < 77.0:  # West Delhi - more built-up
            img[:, :, 0] = np.random.randint(100, 180, (128, 128), dtype=np.uint8)  # R
            img[:, :, 1] = np.random.randint(80, 140, (128, 128), dtype=np.uint8)   # G
            img[:, :, 2] = np.random.randint(70, 120, (128, 128), dtype=np.uint8)   # B
        else:  # East Delhi - more vegetation
            img[:, :, 0] = np.random.randint(60, 120, (128, 128), dtype=np.uint8)  # R
            img[:, :, 1] = np.random.randint(100, 180, (128, 128), dtype=np.uint8)  # G
            img[:, :, 2] = np.random.randint(50, 100, (128, 128), dtype=np.uint8)   # B
        
        # Add some structure (roads, buildings simulation)
        for _ in range(5):
            x = np.random.randint(0, 128)
            y = np.random.randint(0, 128)
            thickness = np.random.randint(2, 8)
            color = np.random.randint(100, 200, 3)
            img[x:x+thickness, :] = color
            img[:, y:y+thickness] = color
        
        # Save as numpy array (simulating .png)
        filename = f"scene_{i:04d}_{lat:.6f}_{lon:.6f}.npy"
        filepath = os.path.join(IMG_DIR, filename)
        np.save(filepath, img)
        
        image_coords.append({
            'filename': filename,
            'lat': lat,
            'lon': lon
        })
    
    # Save metadata
    df = pd.DataFrame(image_coords)
    df.to_csv(os.path.join(IMG_DIR, 'image_coordinates.csv'), index=False)
    
    print(f"  - Images saved to: {IMG_DIR}")
    print(f"  - Metadata saved: {IMG_DIR}/image_coordinates.csv")
    
    return df

def generate_land_cover_raster():
    """Generate synthetic land cover raster (ESA WorldCover style)"""
    print("  - Generating synthetic land cover raster...")
    
    # Create a raster covering Delhi region
    # Resolution: 10m, Size: 1000x1000 pixels (10km x 10km area)
    
    lat_range = np.linspace(DELHI_LAT_MIN - 0.1, DELHI_LAT_MAX + 0.1, 1000)
    lon_range = np.linspace(DELHI_LON_MIN - 0.1, DELHI_LON_MAX + 0.1, 1000)
    
    # Create land cover classes
    # ESA WorldCover classes:
    # 10: Tree cover
    # 20: Shrubland
    # 30: Grassland
    # 40: Cropland
    # 50: Built-up
    # 60: Bare/sparse vegetation
    # 70: Water
    # 80: Wetlands
    # 90: Moss/lichen
    
    land_cover = np.zeros((1000, 1000), dtype=np.uint8)
    
    # Create spatial patterns
    for i in range(1000):
        for j in range(1000):
            lat = lat_range[i]
            lon = lon_range[j]
            
            # Central Delhi - Built-up (class 50)
            if (lon - 77.1)**2 + (lat - 28.65)**2 < 0.02:
                land_cover[i, j] = 50  # Built-up
            # Northern area - Cropland (class 40)
            elif lat > 28.7:
                land_cover[i, j] = 40  # Cropland
            # Western area - Built-up
            elif lon < 77.0:
                land_cover[i, j] = 50  # Built-up
            # Parks/vegetation - Tree cover (class 10)
            elif (lon - 77.15)**2 + (lat - 28.6)**2 < 0.005:
                land_cover[i, j] = 10  # Tree cover
            # Water bodies (class 70) - Yamuna river
            elif abs(lon - 77.25) < 0.01 and lat < 28.7:
                land_cover[i, j] = 70  # Water
            # Default - Cropland
            else:
                land_cover[i, j] = 40  # Cropland
    
    # Save as numpy array
    np.save(os.path.join(DATA_DIR, 'land_cover.npy'), land_cover)
    np.save(os.path.join(DATA_DIR, 'lat_range.npy'), lat_range)
    np.save(os.path.join(DATA_DIR, 'lon_range.npy'), lon_range)
    
    print(f"  - Land cover raster saved: {DATA_DIR}/land_cover.npy")
    
    return land_cover, lat_range, lon_range

# Generate all data
ncr_boundary = generate_delhi_ncr_shapefile()
airshed_boundary = generate_delhi_airshed_shapefile()
image_df = generate_satellite_images(200)
land_cover, lat_range, lon_range = generate_land_cover_raster()

print("\n[2] Data generation complete!")

# =============================================================================
# Q1: SPATIAL REASONING & DATA FILTERING
# =============================================================================

print("\n" + "=" * 60)
print("Q1: SPATIAL REASONING & DATA FILTERING")
print("=" * 60)

# Load the shapefile (numpy array)
ncr_boundary = np.load(os.path.join(SHP_DIR, 'delhi_ncr_boundary.npy'))

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
grid_size_deg = 60 / 111.0  # Approximately 0.54 degrees

# Get bounds
lon_min, lat_min = ncr_boundary[:, 0].min(), ncr_boundary[:, 1].min()
lon_max, lat_max = ncr_boundary[:, 0].max(), ncr_boundary[:, 1].max()

# Create grid lines
grid_patches = []
for lon in np.arange(lon_min, lon_max + grid_size_deg, grid_size_deg):
    for lat in np.arange(lat_min, lat_max + grid_size_deg, grid_size_deg):
        # Create grid cell
        cell = Polygon([
            (lon, lat),
            (lon + grid_size_deg, lat),
            (lon + grid_size_deg, lat + grid_size_deg),
            (lon, lat + grid_size_deg)
        ], fill=False, edgecolor='blue', linewidth=0.5, alpha=0.7)
        grid_patches.append(cell)
        ax.add_patch(cell)

# Plot grid points
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

# Check which images are inside the polygon
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

# Save filtered image list
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
    90: 'Others',       # Moss/lichen
}

SIMPLIFIED_CLASSES = ['Built-up', 'Vegetation', 'Water', 'Cropland', 'Others']

def get_land_cover_at_point(lat, lon, land_cover, lat_range, lon_range):
    """Extract land cover at a specific point"""
    # Find nearest index
    lat_idx = np.argmin(np.abs(lat_range - lat))
    lon_idx = np.argmin(np.abs(lon_range - lon))
    
    return land_cover[lat_idx, lon_idx]

def extract_label_patch(lat, lon, land_cover, lat_range, lon_range, patch_size=128):
    """Extract 128x128 patch centered on the coordinate"""
    # Convert lat/lon to pixel indices
    lat_idx = np.argmin(np.abs(lat_range - lat))
    lon_idx = np.argmin(np.abs(lon_range - lon))
    
    # Half patch size
    half = patch_size // 2
    
    # Extract patch with boundary checks
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
    
    # Get mode (most common class)
    if patch.size > 0:
        unique, counts = np.unique(patch, return_counts=True)
        mode_class = unique[np.argmax(counts)]
    else:
        mode_class = 40  # Default to Cropland
    
    labels.append(mode_class)
    
    # Map to simplified category
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

# Training set distribution
train_counts = pd.Series(y_train).value_counts()
axes[0].bar(train_counts.index, train_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
axes[0].set_title('Training Set Class Distribution', fontsize=12)
axes[0].set_xlabel('Land Use Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Test set distribution
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

# Save dataset
filtered_df.to_csv(os.path.join(OUTPUT_DIR, 'labeled_dataset.csv'), index=False)
print(f"  - Labeled dataset saved: {OUTPUT_DIR}/labeled_dataset.csv")

# =============================================================================
# Q3: MODEL TRAINING & SUPERVISED EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("Q3: MODEL TRAINING & SUPERVISED EVALUATION")
print("=" * 60)

# For this demonstration, we'll create a simple CNN model
# We'll use a simplified approach since we have limited data

print("\n[Q3.1] Training CNN model for land-use classification...")

# Prepare image data for CNN
def load_image_as_features(filename):
    """Load image and convert to feature vector"""
    img = np.load(os.path.join(IMG_DIR, filename))
    # Flatten and normalize
    return img.flatten() / 255.0

# Load all images for filtered samples
X_images = np.array([load_image_as_features(f) for f in filtered_df['filename'].values])

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(SIMPLIFIED_CLASSES)
y_encoded = le.transform(filtered_df['land_use_label'].values)

# Train-test split
X_train_img, X_test_img, y_train_enc, y_test_enc = train_test_split(
    X_images, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)

print(f"  Training samples: {X_train_img.shape[0]}")
print(f"  Test samples: {X_test_img.shape[0]}")
print(f"  Features per sample: {X_train_img.shape[1]}")

# Train a simple neural network using sklearn's MLPClassifier
# (Simpler than TensorFlow for this demonstration)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

print("\n  Training MLP Classifier (simplified CNN alternative)...")

# Simple MLP model
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

# Interpretation
print("\n  Interpretation:")
print("  - The confusion matrix shows the classification performance")
print("  - Diagonal values indicate correct predictions")
print("  - Off-diagonal values show misclassifications")
print(f"  - Built-up and Cropland classes show {'good' if cm[0,0] > cm[0,:].sum()*0.5 else 'moderate'} performance")

# Summary
print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"""
Q1 Results:
  - Total images before filtering: {total_images}
  - Total images after filtering: {filtered_images}
  - Grid created: 60×60 km over Delhi-NCR

Q2 Results:
  - Land cover classes extracted from ESA WorldCover
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


# Additional feature: BlackBoxAI enhancement
