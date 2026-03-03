"""
Q1: Spatial Reasoning & Data Filtering
Proper implementation with UTM coordinates (EPSG:32644)

This script:
- Uses EPSG:32644 (UTM Zone 44N) for metric grid operations
- Creates 60km x 60km grid cells
- Filters satellite patches inside Delhi-NCR region
"""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import pandas as pd
import numpy as np
import os
import sys

# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("output")

# Create directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Q1: SPATIAL REASONING & DATA FILTERING")
print("=" * 60)

# Try to import geopandas for proper shapefile handling
HAS_GEOPANDAS = False
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, Point
    HAS_GEOPANDAS = True
    print("Using geopandas for geospatial operations")
except ImportError:
    print("Note: geopandas not available, using fallback method")

# Delhi-NCR boundary coordinates (EPSG:4326 - WGS84)
DELHI_LAT_MIN, DELHI_LAT_MAX = 28.4, 28.9
DELHI_LON_MIN, DELHI_LON_MAX = 76.8, 77.5

# Point in polygon function
def point_in_polygon(x, y, polygon):
    """Check if point (x, y) is inside polygon (ray casting algorithm)"""
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

# UTM Zone 44N conversion (approximate for Delhi region)
def latlon_to_utm(lon, lat):
    """Convert WGS84 to UTM Zone 44N (EPSG:32644)"""
    # Central meridian: 75°E
    utm_x = (lon - 75) * 111320 * np.cos(np.radians(28.65))
    utm_y = lat * 110540
    return utm_x, utm_y

def utm_to_latlon(utm_x, utm_y):
    """Convert UTM Zone 44N to WGS84"""
    lon = utm_x / (111320 * np.cos(np.radians(28.65))) + 75
    lat = utm_y / 110540
    return lon, lat

# Get UTM bounds
utm_minx, utm_miny = latlon_to_utm(DELHI_LON_MIN, DELHI_LAT_MIN)
utm_maxx, utm_maxy = latlon_to_utm(DELHI_LON_MAX, DELHI_LAT_MAX)

print(f"\n[Q1.1] Delhi-NCR Bounds:")
print(f"  Lat/Lon: ({DELHI_LAT_MIN}, {DELHI_LON_MIN}) to ({DELHI_LAT_MAX}, {DELHI_LON_MAX})")
print(f"  UTM: ({utm_minx:.0f}, {utm_miny:.0f}) to ({utm_maxx:.0f}, {utm_maxy:.0f})")

# Build 60km grid in UTM
grid_size_m = 60000  # 60km in meters
grid_patches = []

for x in np.arange(utm_minx, utm_maxx, grid_size_m):
    for y in np.arange(utm_miny, utm_maxy, grid_size_m):
        cell = [
            (x, y),
            (x + grid_size_m, y),
            (x + grid_size_m, y + grid_size_m),
            (x, y + grid_size_m)
        ]
        grid_patches.append(cell)

print(f"  Created {len(grid_patches)} grid cells (60km x 60km)")

# Check for real data
ncr_geojson_path = RAW_DATA_DIR / "delhi_ncr" / "delhi_ncr_region.geojson"
patch_folder = RAW_DATA_DIR / "sentinel_patches"

if ncr_geojson_path.exists() and HAS_GEOPANDAS:
    # Use geopandas for proper shapefile handling
    print(f"\n[Q1.1] Loading Delhi-NCR shapefile from {ncr_geojson_path}")
    gdf = gpd.read_file(str(ncr_geojson_path))
    gdf_region_utm = gdf.to_crs(epsg=32644)
    region_geom_utm = gdf_region_utm.geometry.union_all()
    
    # Filter grid cells that intersect with NCR
    grid_cells = [Polygon(cell) for cell in grid_patches]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32644")
    grid_gdf = grid_gdf[grid_gdf.intersects(region_geom_utm)]
    print(f"  Grid cells touching NCR: {len(grid_gdf)}")
    
    # List and filter patches
    patch_files = list(patch_folder.glob("*.png"))
    print(f"\n[Q1.2] Total images before filtering: {len(patch_files)}")
    
    # Parse and filter
    records = []
    for p in patch_files:
        parts = p.stem.split("_")
        if len(parts) != 2:
            continue
        lat, lon = float(parts[0]), float(parts[1])
        utm_x, utm_y = latlon_to_utm(lon, lon)
        
        if region_geom_utm.contains(Point(utm_x, utm_y)):
            records.append({"filename": p.name, "lat": lat, "lon": lon})
    
    filtered_df = pd.DataFrame(records)
    print(f"  Total images after filtering: {len(filtered_df)}")
    
else:
    # Use existing .npy images from data/images
    image_dir = Path("data/images")
    patch_files = list(image_dir.glob("*.npy"))
    print(f"\n[Q1.2] Found {len(patch_files)} satellite images in data/images")
    
    # Parse filenames - format: scene_0000_28.574540_77.334641.npy
    records = []
    for p in patch_files:
        parts = p.stem.split("_")
        if len(parts) >= 4:
            try:
                lat = float(parts[-2])
                lon = float(parts[-1])
                records.append({"filename": p.name, "lat": lat, "lon": lon})
            except:
                continue
    
    # Filter using point-in-polygon (rectangle)
    boundary = [
        (DELHI_LON_MIN, DELHI_LAT_MIN),
        (DELHI_LON_MAX, DELHI_LAT_MIN),
        (DELHI_LON_MAX, DELHI_LAT_MAX),
        (DELHI_LON_MIN, DELHI_LAT_MAX)
    ]
    
    filtered_records = []
    for rec in records:
        lat, lon = rec['lat'], rec['lon']
        if point_in_polygon(lon, lat, boundary):
            filtered_records.append(rec)
    
    filtered_df = pd.DataFrame(filtered_records)
    print(f"  Total images after filtering: {len(filtered_df)}")

# Save filtered images
filtered_df.to_csv(PROCESSED_DATA_DIR / "filtered_patches.csv", index=False)
filtered_df.to_csv(OUTPUT_DIR / "filtered_images.csv", index=False)
print(f"\n[Q1.3] Saved filtered images to:")
print(f"  - {PROCESSED_DATA_DIR / 'filtered_patches.csv'}")
print(f"  - {OUTPUT_DIR / 'filtered_images.csv'}")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot boundary
boundary_coords = [
    (DELHI_LON_MIN, DELHI_LAT_MIN),
    (DELHI_LON_MAX, DELHI_LAT_MIN),
    (DELHI_LON_MAX, DELHI_LAT_MAX),
    (DELHI_LON_MIN, DELHI_LAT_MAX),
    (DELHI_LON_MIN, DELHI_LAT_MIN)
]
boundary_poly = MplPolygon(boundary_coords, fill=False, edgecolor='red', linewidth=2, label='Delhi-NCR Boundary')
ax.add_patch(boundary_poly)

# Plot grid (convert UTM to lat/lon for plotting)
for cell in grid_patches:
    cell_lonlat = [utm_to_latlon(x, y) for x, y in cell]
    cell_lon = [p[0] for p in cell_lonlat]
    cell_lat = [p[1] for p in cell_lonlat]
    grid_poly = MplPolygon(list(zip(cell_lon, cell_lat)), fill=False, edgecolor='blue', 
                          linewidth=0.5, alpha=0.7)
    ax.add_patch(grid_poly)

# Plot filtered points
if len(filtered_df) > 0:
    ax.scatter(filtered_df['lon'], filtered_df['lat'], c='green', s=10, 
              label=f'Filtered Patches ({len(filtered_df)})', zorder=5)

ax.set_xlim(DELHI_LON_MIN - 0.1, DELHI_LON_MAX + 0.1)
ax.set_ylim(DELHI_LAT_MIN - 0.1, DELHI_LAT_MAX + 0.1)
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Delhi-NCR with 60×60 km Grid (EPSG:4326)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "q1_delhi_ncr_grid.png", dpi=150)
plt.close()
print(f"\n[Q1.3] Plot saved: {OUTPUT_DIR / 'q1_delhi_ncr_grid.png'}")

print("\n" + "=" * 60)
print("Q1 COMPLETE")
print("=" * 60)
