"""
Data loading module for EV charging infrastructure optimization
Loads all GeoJSON and CSV files, validates CRS, and returns standardized formats
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Import config and logger
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *
from utils.logger import setup_logger, log_section, log_data_summary

logger = setup_logger(__name__)


class DataLoader:
    """Handles loading and initial validation of all project data files"""
    
    def __init__(self, target_crs: str = TARGET_CRS):
        """
        Initialize DataLoader
        
        Args:
            target_crs: Target coordinate reference system (default from config)
        """
        self.target_crs = target_crs
        logger.info(f"DataLoader initialized with target CRS: {target_crs}")
    
    def load_bbmp_wards(self) -> gpd.GeoDataFrame:
        """
        Load BBMP ward boundaries
        
        Returns:
            GeoDataFrame with ward polygons
        """
        log_section(logger, "Loading BBMP Ward Boundaries")
        
        if not BBMP_WARDS_FILE.exists():
            raise FileNotFoundError(f"BBMP wards file not found: {BBMP_WARDS_FILE}")
        
        wards = gpd.read_file(BBMP_WARDS_FILE)
        logger.info(f"Loaded {len(wards)} wards")
        
        # Validate CRS and reproject if needed
        wards = self._validate_and_reproject(wards, "BBMP Wards")
        
        log_data_summary(logger, wards, "BBMP Wards")
        return wards
    
    def load_traffic_data(self) -> gpd.GeoDataFrame:
        """
        Load and combine all traffic GeoJSON files
        
        Returns:
            Combined GeoDataFrame with all traffic segments
        """
        log_section(logger, "Loading Traffic Data")
        
        traffic_files = list(TRAFFIC_DIR.glob("*.geojson"))
        
        if not traffic_files:
            raise FileNotFoundError(f"No traffic files found in {TRAFFIC_DIR}")
        
        logger.info(f"Found {len(traffic_files)} traffic GeoJSON files")
        
        traffic_dfs = []
        for file in traffic_files:
            try:
                gdf = gpd.read_file(file)
                gdf['source_file'] = file.name
                traffic_dfs.append(gdf)
                logger.debug(f"Loaded {file.name}: {len(gdf)} features")
            except Exception as e:
                logger.warning(f"Failed to load {file.name}: {e}")
        
        # Combine all traffic data
        traffic = pd.concat(traffic_dfs, ignore_index=True)
        traffic = gpd.GeoDataFrame(traffic, crs=traffic_dfs[0].crs)
        
        logger.info(f"Combined traffic data: {len(traffic)} total segments")
        
        # Validate and reproject
        traffic = self._validate_and_reproject(traffic, "Traffic Data")
        
        log_data_summary(logger, traffic, "Traffic Data")
        return traffic
    
    def load_grid_substations(self) -> pd.DataFrame:
        """
        Load grid substation data from CSV
        
        Returns:
            DataFrame with substation information
        """
        log_section(logger, "Loading Grid Substations")
        
        if not GRID_SUBSTATIONS_FILE.exists():
            raise FileNotFoundError(f"Grid file not found: {GRID_SUBSTATIONS_FILE}")
        
        grid = pd.read_csv(GRID_SUBSTATIONS_FILE)
        logger.info(f"Loaded {len(grid)} substations")
        
        log_data_summary(logger, grid, "Grid Substations")
        return grid
    
    def load_existing_chargers(self) -> gpd.GeoDataFrame:
        """
        Load existing charger locations from OpenChargeMap
        
        Returns:
            GeoDataFrame with charger points
        """
        log_section(logger, "Loading Existing Chargers")
        
        if not OPENCHARGEMAP_FILE.exists():
            raise FileNotFoundError(f"Chargers file not found: {OPENCHARGEMAP_FILE}")
        
        chargers = gpd.read_file(OPENCHARGEMAP_FILE)
        logger.info(f"Loaded {len(chargers)} existing chargers")
        
        # Validate and reproject
        chargers = self._validate_and_reproject(chargers, "Existing Chargers")
        
        log_data_summary(logger, chargers, "Existing Chargers")
        return chargers
    
    def load_landuse_zones(self) -> gpd.GeoDataFrame:
        """
        Load land use zone data
        
        Returns:
            GeoDataFrame with land use polygons
        """
        log_section(logger, "Loading Land Use Zones")
        
        if not LANDUSE_ZONES_FILE.exists():
            raise FileNotFoundError(f"Land use file not found: {LANDUSE_ZONES_FILE}")
        
        landuse = gpd.read_file(LANDUSE_ZONES_FILE)
        logger.info(f"Loaded {len(landuse)} land use zones")
        
        # Validate and reproject
        landuse = self._validate_and_reproject(landuse, "Land Use Zones")
        
        log_data_summary(logger, landuse, "Land Use Zones")
        return landuse
    
    def load_cost_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all cost-related CSV files
        
        Returns:
            Dictionary of DataFrames: {'tariffs', 'installation', 'operational', 'land_rental'}
        """
        log_section(logger, "Loading Cost Data")
        
        cost_data = {}
        
        # Tariffs
        cost_data['tariffs'] = pd.read_csv(TARIFFS_FILE, encoding='utf-8')
        logger.info(f"Loaded tariffs: {len(cost_data['tariffs'])} rows")
        
        # Installation costs
        cost_data['installation'] = pd.read_csv(INSTALLATION_COSTS_FILE)
        logger.info(f"Loaded installation costs: {len(cost_data['installation'])} charger types")
        
        # Operational costs
        cost_data['operational'] = pd.read_csv(OPERATIONAL_COSTS_FILE, on_bad_lines='warn')
        logger.info(f"Loaded operational costs: {len(cost_data['operational'])} rows")
        
        # Land rental rates
        cost_data['land_rental'] = pd.read_csv(LAND_RENTAL_FILE)
        logger.info(f"Loaded land rental rates: {len(cost_data['land_rental'])} zone types")
        
        return cost_data
    
    def load_fleet_data(self) -> pd.DataFrame:
        """
        Load EV fleet specifications
        
        Returns:
            DataFrame with EV models and specs
        """
        log_section(logger, "Loading EV Fleet Data")
        
        if not EV_FLEET_FILE.exists():
            raise FileNotFoundError(f"Fleet file not found: {EV_FLEET_FILE}")
        
        fleet = pd.read_csv(EV_FLEET_FILE)
        logger.info(f"Loaded {len(fleet)} EV models")
        
        log_data_summary(logger, fleet, "EV Fleet")
        return fleet
    
    def load_all_data(self) -> Dict:
        """
        Load all project data files at once
        
        Returns:
            Dictionary with all loaded datasets
        """
        log_section(logger, "LOADING ALL PROJECT DATA", char="*")
        
        data = {
            'wards': self.load_bbmp_wards(),
            'traffic': self.load_traffic_data(),
            'grid': self.load_grid_substations(),
            'chargers': self.load_existing_chargers(),
            'landuse': self.load_landuse_zones(),
            'costs': self.load_cost_data(),
            'fleet': self.load_fleet_data()
        }
        
        log_section(logger, "DATA LOADING COMPLETE", char="*")
        return data
    
    def _validate_and_reproject(self, gdf: gpd.GeoDataFrame, name: str) -> gpd.GeoDataFrame:
        """
        Validate CRS and reproject to target CRS if needed
        
        Args:
            gdf: GeoDataFrame to validate
            name: Name of dataset (for logging)
        
        Returns:
            Reprojected GeoDataFrame
        """
        if gdf.crs is None:
            logger.warning(f"{name}: No CRS defined, assuming {self.target_crs}")
            gdf = gdf.set_crs(self.target_crs)
        elif gdf.crs.to_string() != self.target_crs:
            logger.info(f"{name}: Reprojecting from {gdf.crs} to {self.target_crs}")
            gdf = gdf.to_crs(self.target_crs)
        else:
            logger.info(f"{name}: CRS already {self.target_crs}")
        
        return gdf


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    data = loader.load_all_data()
    
    print("\n" + "="*60)
    print("Data Loading Test Complete")
    print("="*60)
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {len(subvalue)} rows")
        else:
            print(f"{key}: {len(value)} rows")
