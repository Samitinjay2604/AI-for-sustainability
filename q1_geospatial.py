"""
Q1: Spatial Reasoning & Data Filtering
Using proper Geopandas for Delhi-NCR shapefile processing

This script uses:
- geopandas for shapefile handling
- EPSG:32644 (UTM Zone 44N) for metric grid operations
- Proper coordinate transformations
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
import os

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

# Check if GeoJSON file exists
ncr_geojson_path = RAW_DATA_DIR / "delhi_ncr" / "delhi_ncr_region.geojson"

if ncr_geojson_path.exists():
    print(f"\n[Q1.1] Loading Delhi-NCR shapefile from {ncr_geojson_path}")
    
    # Load the shapefile
    gdf = gpd.read_file(str(ncr_geojson_path))
    gdf_region = gdf
    print(gdf_region.head())
    print(f"CRS: {gdf_region.crs}")
    
    # If CRS missing, set it (GeoJSON usually is EPSG:4326)
    if gdf_region.crs is None:
        gdf_region = gdf_region.set_crs(epsg=4326)
    
    # Convert to UTM (meters) for grid + filtering
    gdf_region_utm = gdf_region.to_crs(epsg=32644)
    print(f"Bounds (UTM): {gdf_region_utm.total_bounds}")
    
    # Single merged geometry for point-in-polygon
    region_geom_utm = gdf_region_utm.geometry.union_all()
    
    # Build 60km grid in UTM
    minx, miny, maxx, maxy = gdf_region_utm.total_bounds
    grid_cells = []
    for x in range(int(minx), int(maxx), 60000):
        for y in range(int(miny), int(maxy), 60000):
            cell = Polygon([(x, y), (x + 60000, y), (x + 60000, y + 60000), (x, y + 60000)])
            grid_cells.append(cell)
    
    # Create grid with correct CRS directly
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32644")
    
    # Keep only grid cells that touch NCR (clean plot)
    grid = grid[grid.intersects(region_geom_utm)]
    print(f"\n[Q1.1] Created {len(grid)} grid cells (60km x 60km)")
    
    # List all png patches
    patch_folder = RAW_DATA_DIR / "sentinel_patches"
    
    if patch_folder.exists():
        patch_files = list(patch_folder.glob("*.png"))
        print(f"\n[Q1.2] Total images before filtering: {len(patch_files)}")
        
        # Parse filenames -> points
        records = []
        geoms = []
        for p in patch_files:
            parts = p.stem.split("_")
            if len(parts) != 2:
                continue
            
            lat = float(parts[0])
            lon = float(parts[1])
            
            records.append({"filename": p.name, "lat": lat, "lon": lon})
            geoms.append(Point(lon, lat))
        
        points_gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
        points_utm = points_gdf.to_crs(epsg=32644)
        
        # Filter INSIDE region
        mask = points_utm.within(region_geom_utm)
        filtered_points_utm = points_utm[mask].copy()
        
        print(f"Total images after filtering: {len(filtered_points_utm)}")
        
        # Save CSV for Q2 (lat/lon still present as columns)
        filtered_points_utm[["filename", "lat", "lon"]].to_csv(
            PROCESSED_DATA_DIR / "filtered_patches.csv", index=False
        )
        print(f"Saved: {PROCESSED_DATA_DIR / 'filtered_patches.csv'}")
        
        # Also save to output folder for consistency
        filtered_points_utm[["filename", "lat", "lon"]].to_csv(
            OUTPUT_DIR / "filtered_images.csv", index=False
        )
    else:
        print(f"\n[Q1.2] No sentinel patches found in {patch_folder}")
        print("Using synthetic data fallback...")
        
        # Generate synthetic data
        np.random.seed(42)
        num_images = 200
        
        # Delhi-NCR bounding box
        lat_min, lat_max = 28.4, 28.9
        lon_min, lon_max = 76.8, 77.5
        
        lats = np.random.uniform(lat_min, lat_max, num_images)
        lons = np.random.uniform(lon_min, lon_max, num_images)
        
        records = []
        geoms = []
        for i in range(num_images):
            lat, lon = lats[i], lons[i]
            records.append({"filename": f"scene_{i:04d}_{lat:.6f}_{lon:.6f}.png", "lat": lat, "lon": lon})
            geoms.append(Point(lon, lat))
        
        points_gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
        points_utm = points_gdf.to_crs(epsg=32644)
        
        # Filter INSIDE region
        mask = points_utm.within(region_geom_utm)
        filtered_points_utm = points_utm[mask].copy()
        
        print(f"Total images after filtering: {len(filtered_points_utm)}")
        
        # Save CSV
        filtered_points_utm[["filename", "lat", "lon"]].to_csv(
            PROCESSED_DATA_DIR / "filtered_patches.csv", index=False
        )
        filtered_points_utm[["filename", "lat", "lon"]].to_csv(
            OUTPUT_DIR / "filtered_images.csv", index=False
        )
    
    # Plot (all in EPSG:32644)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_region_utm.boundary.plot(ax=ax, edgecolor="red", linewidth=1, label="Delhi-NCR Boundary")
    grid.boundary.plot(ax=ax, edgecolor="black", linewidth=0.5, label="60km Grid")
    filtered_points_utm.plot(ax=ax, markersize=3, color="blue", label="Filtered Patches")
    
    # Zoom to grid bounds
    gxmin, gymin, gxmax, gymax = grid.total_bounds
    ax.set_xlim(gxmin, gxmax)
    ax.set_ylim(gymin, gymax)
    
    plt.title("Delhi-NCR with 60×60 km Grid (EPSG:32644)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "q1_delhi_ncr_grid.png", dpi=150)
    plt.close()
    print(f"\n[Q1.3] Plot saved: {OUTPUT_DIR / 'q1_delhi_ncr_grid.png'}")
    
else:
    print(f"\nGeoJSON file not found: {ncr_geojson_path}")
    print("Please download from: https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed")
    print("\nUsing fallback mode with synthetic boundary...")
    
    # Fallback: Create synthetic boundary
    from delhi_airshed_pipeline import *
    
    print("\n[Q1.1-3] Using synthetic data from main pipeline")
    print("Run 'python delhi_airshed_pipeline.py' for complete pipeline")

print("\n" + "=" * 60)
print("Q1 COMPLETE")
print("=" * 60)
