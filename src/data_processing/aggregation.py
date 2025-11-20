"""
Ward-level aggregation module

Aggregates spatially joined datasets to ward level for optimization and analysis.
Computes summary statistics for traffic, charging demand, grid capacity, and land use feasibility.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import *
from src.utils.logger import setup_logger, log_section, log_data_summary

logger = setup_logger(__name__)


class WardAggregator:
    """Aggregates spatial join results to ward-level metrics"""
    
    def __init__(self, ward_id_col: str = 'KGISWardID'):
        """
        Initialize aggregator
        
        Args:
            ward_id_col: Column name for ward identifier
        """
        self.ward_id_col = ward_id_col
        logger.info(f"WardAggregator initialized with ward ID column: '{ward_id_col}'")
    
    def aggregate_traffic(self, traffic_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate traffic data to ward level
        
        Metrics computed:
        - total_traffic_segments: Count of road segments
        - total_road_length_km: Sum of segment distances
        - avg_speed_limit: Average speed limit
        - avg_frc: Average functional road class (1=motorway, 8=local)
        - unique_streets: Number of unique street names
        
        Args:
            traffic_gdf: Traffic data with ward assignments
        
        Returns:
            DataFrame with ward-level traffic metrics
        """
        log_section(logger, "Aggregating Traffic Data")
        
        # Filter to matched segments only
        matched = traffic_gdf[traffic_gdf[self.ward_id_col].notna()].copy()
        logger.info(f"Processing {len(matched):,} matched traffic segments")
        
        # Aggregate by ward
        agg_dict = {
            'segmentId': 'count',  # Count segments
            'distance': 'sum',     # Total road length
            'speedLimit': 'mean',  # Average speed
            'frc': 'mean',         # Average road class
            'streetName': 'nunique'  # Unique streets
        }
        
        ward_traffic = matched.groupby(self.ward_id_col).agg(agg_dict).reset_index()
        
        # Rename columns
        ward_traffic.columns = [
            self.ward_id_col,
            'total_traffic_segments',
            'total_road_length_m',
            'avg_speed_limit_kmh',
            'avg_frc',
            'unique_streets'
        ]
        
        # Convert distance to km
        ward_traffic['total_road_length_km'] = ward_traffic['total_road_length_m'] / 1000
        ward_traffic = ward_traffic.drop(columns=['total_road_length_m'])
        
        # Round metrics
        ward_traffic['total_road_length_km'] = ward_traffic['total_road_length_km'].round(2)
        ward_traffic['avg_speed_limit_kmh'] = ward_traffic['avg_speed_limit_kmh'].round(1)
        ward_traffic['avg_frc'] = ward_traffic['avg_frc'].round(1)
        
        logger.info(f"Aggregated traffic for {len(ward_traffic)} wards")
        logger.info(f"Total road network: {ward_traffic['total_road_length_km'].sum():.1f} km")
        
        return ward_traffic
    
    def aggregate_chargers(self, chargers_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Aggregate existing chargers to ward level
        
        Metrics computed:
        - existing_chargers_count: Number of charging stations
        - total_charging_points: Sum of numberOfPoints
        - public_chargers: Count of public chargers
        - private_chargers: Count of private chargers
        
        Args:
            chargers_gdf: Charger data with ward assignments
        
        Returns:
            DataFrame with ward-level charger metrics
        """
        log_section(logger, "Aggregating Existing Chargers")
        
        # Filter to matched chargers
        matched = chargers_gdf[chargers_gdf[self.ward_id_col].notna()].copy()
        logger.info(f"Processing {len(matched)} matched chargers")
        
        # Create usage type flags
        matched['is_public'] = (matched['usageType'].str.lower() == 'public').astype(int)
        matched['is_private'] = (matched['usageType'].str.lower() == 'private').astype(int)
        
        # Aggregate by ward
        agg_dict = {
            'id': 'count',
            'numberOfPoints': 'sum',
            'is_public': 'sum',
            'is_private': 'sum'
        }
        
        ward_chargers = matched.groupby(self.ward_id_col).agg(agg_dict).reset_index()
        
        # Rename columns
        ward_chargers.columns = [
            self.ward_id_col,
            'existing_chargers_count',
            'total_charging_points',
            'public_chargers',
            'private_chargers'
        ]
        
        # Fill NaN with 0
        ward_chargers['total_charging_points'] = ward_chargers['total_charging_points'].fillna(0).astype(int)
        
        logger.info(f"Aggregated chargers for {len(ward_chargers)} wards")
        logger.info(f"Total existing chargers: {ward_chargers['existing_chargers_count'].sum()}")
        logger.info(f"Total charging points: {ward_chargers['total_charging_points'].sum()}")
        
        return ward_chargers
    
    def aggregate_grid(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate grid substations to ward level
        
        Metrics computed:
        - substation_count: Number of substations
        - avg_voltage_kv: Average voltage class
        - high_voltage_substations: Count of substations >=110kV
        - matched_substations: Count with confirmed location
        
        Args:
            grid_df: Grid data with ward assignments
        
        Returns:
            DataFrame with ward-level grid metrics
        """
        log_section(logger, "Aggregating Grid Substations")
        
        # Filter to matched substations
        matched = grid_df[grid_df[self.ward_id_col].notna()].copy()
        logger.info(f"Processing {len(matched)} substations")
        
        # Create high voltage flag (>=110kV is transmission level)
        matched['is_high_voltage'] = (matched['Voltage Class (in kV)'] >= 110).astype(int)
        
        # Create confirmed match flag
        matched['is_confirmed'] = (matched['match_method'] == 'name_match').astype(int)
        
        # Aggregate by ward
        agg_dict = {
            '_id': 'count',
            'Voltage Class (in kV)': 'mean',
            'is_high_voltage': 'sum',
            'is_confirmed': 'sum'
        }
        
        ward_grid = matched.groupby(self.ward_id_col).agg(agg_dict).reset_index()
        
        # Rename columns
        ward_grid.columns = [
            self.ward_id_col,
            'substation_count',
            'avg_voltage_kv',
            'high_voltage_substations',
            'matched_substations'
        ]
        
        # Round voltage
        ward_grid['avg_voltage_kv'] = ward_grid['avg_voltage_kv'].round(1)
        
        logger.info(f"Aggregated grid for {len(ward_grid)} wards")
        logger.info(f"Total substations: {ward_grid['substation_count'].sum()}")
        logger.info(f"High voltage (>=110kV): {ward_grid['high_voltage_substations'].sum()}")
        
        return ward_grid
    
    def compute_land_use_feasibility(self, landuse_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute land use feasibility scores for each ward
        
        Scoring logic:
        - Commercial zones: High feasibility (score 1.0)
        - Mixed use zones: Medium-high feasibility (score 0.8)
        - Residential zones: Medium feasibility (score 0.6)
        - Industrial zones: Low-medium feasibility (score 0.4)
        - Green/restricted zones: Low feasibility (score 0.2)
        
        Args:
            landuse_df: Ward-level land use percentage data
        
        Returns:
            DataFrame with feasibility scores
        """
        log_section(logger, "Computing Land Use Feasibility")
        
        # Define zone feasibility weights
        zone_weights = {
            'BOMMANAHALLI': 0.8,    # Mixed commercial/residential
            'DASARAHALLI': 0.6,     # Primarily residential
            'EAST': 0.7,            # Mixed zones
            'MAHADEVAPURA': 0.9,    # Tech/commercial hub
            'RR NAGARA': 0.5,       # Residential/industrial mix
            'SOUTH': 0.8,           # Commercial/institutional
            'WEST': 0.7,            # Mixed development
            'YELAHANKA': 0.6        # Residential/airport area
        }
        
        logger.info(f"Zone feasibility weights: {zone_weights}")
        
        # Create feasibility dataframe
        feasibility_df = landuse_df[[self.ward_id_col]].copy()
        
        # Compute weighted feasibility score
        feasibility_df['land_use_feasibility_score'] = 0.0
        
        for zone, weight in zone_weights.items():
            if zone in landuse_df.columns:
                # Weighted sum: zone_percentage * zone_weight
                feasibility_df['land_use_feasibility_score'] += (
                    landuse_df[zone].fillna(0) * weight
                )
        
        # Normalize to 0-1 range (percentages are already 0-100, so divide by 100)
        feasibility_df['land_use_feasibility_score'] = (
            feasibility_df['land_use_feasibility_score'] / 100
        ).round(3)
        
        # Add dominant zone
        zone_cols = [col for col in landuse_df.columns if col not in [self.ward_id_col, 'KGISWardName']]
        feasibility_df['dominant_zone'] = landuse_df[zone_cols].idxmax(axis=1)
        
        logger.info(f"Computed feasibility for {len(feasibility_df)} wards")
        logger.info(f"Average feasibility: {feasibility_df['land_use_feasibility_score'].mean():.3f}")
        
        return feasibility_df
    
    def estimate_charging_demand(self, ward_traffic: pd.DataFrame, fleet_data: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate EV charging demand based on traffic and fleet projections
        
        Simple demand model:
        - Assume EVs per 1000 population (proxy: road segments)
        - Project charging sessions per day based on traffic density
        - Estimate energy demand (kWh/day) based on average EV consumption
        
        Args:
            ward_traffic: Ward-level traffic metrics
            fleet_data: EV fleet specifications
        
        Returns:
            DataFrame with demand estimates
        """
        log_section(logger, "Estimating Charging Demand")
        
        # Simple demand model assumptions
        EV_ADOPTION_RATE = 0.05  # 5% of vehicles are EVs (conservative estimate)
        SESSIONS_PER_DAY_PER_KM = 0.1  # Charging sessions per km of road per day
        AVG_SESSION_ENERGY_KWH = 25  # Average charging session (kWh)
        
        logger.info(f"Demand model assumptions:")
        logger.info(f"  - EV adoption rate: {EV_ADOPTION_RATE*100}%")
        logger.info(f"  - Sessions per km/day: {SESSIONS_PER_DAY_PER_KM}")
        logger.info(f"  - Avg session energy: {AVG_SESSION_ENERGY_KWH} kWh")
        
        demand_df = ward_traffic[[self.ward_id_col, 'total_road_length_km']].copy()
        
        # Estimate daily charging sessions based on road network
        demand_df['estimated_daily_sessions'] = (
            demand_df['total_road_length_km'] * SESSIONS_PER_DAY_PER_KM
        ).round(0).astype(int)
        
        # Estimate daily energy demand (kWh)
        demand_df['estimated_daily_demand_kwh'] = (
            demand_df['estimated_daily_sessions'] * AVG_SESSION_ENERGY_KWH
        ).round(0).astype(int)
        
        # Estimate monthly demand
        demand_df['estimated_monthly_demand_kwh'] = (
            demand_df['estimated_daily_demand_kwh'] * 30
        ).round(0).astype(int)
        
        logger.info(f"Total estimated daily demand: {demand_df['estimated_daily_demand_kwh'].sum():,} kWh")
        logger.info(f"Total estimated monthly demand: {demand_df['estimated_monthly_demand_kwh'].sum():,} kWh")
        
        return demand_df
    
    def aggregate_all(self, joined_data: Dict, wards_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Aggregate all datasets to unified ward-level GeoDataFrame
        
        Args:
            joined_data: Dictionary of spatially joined datasets
            wards_gdf: Original ward boundaries
        
        Returns:
            GeoDataFrame with all ward-level metrics
        """
        log_section(logger, "AGGREGATING ALL DATASETS TO WARD LEVEL", char="*")
        
        # Start with ward geometries
        ward_data = wards_gdf[[self.ward_id_col, 'KGISWardName', 'geometry']].copy()
        logger.info(f"Starting with {len(ward_data)} wards")
        
        # Aggregate each dataset
        traffic_agg = self.aggregate_traffic(joined_data['traffic'])
        chargers_agg = self.aggregate_chargers(joined_data['chargers'])
        grid_agg = self.aggregate_grid(joined_data['grid'])
        feasibility = self.compute_land_use_feasibility(joined_data['landuse'])
        
        # Estimate demand
        demand = self.estimate_charging_demand(traffic_agg, joined_data['fleet'])
        
        # Merge all metrics
        log_section(logger, "Merging All Ward-Level Metrics")
        
        ward_data = ward_data.merge(traffic_agg, on=self.ward_id_col, how='left', suffixes=('', '_drop'))
        logger.info("Merged traffic metrics")
        
        ward_data = ward_data.merge(chargers_agg, on=self.ward_id_col, how='left', suffixes=('', '_drop'))
        logger.info("Merged charger metrics")
        
        ward_data = ward_data.merge(grid_agg, on=self.ward_id_col, how='left', suffixes=('', '_drop'))
        logger.info("Merged grid metrics")
        
        ward_data = ward_data.merge(feasibility, on=self.ward_id_col, how='left', suffixes=('', '_drop'))
        logger.info("Merged feasibility scores")
        
        ward_data = ward_data.merge(demand[[self.ward_id_col, 'estimated_daily_sessions', 'estimated_daily_demand_kwh', 'estimated_monthly_demand_kwh']], 
                                   on=self.ward_id_col, how='left', suffixes=('', '_drop'))
        logger.info("Merged demand estimates")
        
        # Drop duplicate columns
        ward_data = ward_data[[col for col in ward_data.columns if not col.endswith('_drop')]]
        
        # Fill NaN values with 0 for count metrics
        count_cols = [
            'total_traffic_segments', 'unique_streets',
            'existing_chargers_count', 'total_charging_points', 'public_chargers', 'private_chargers',
            'substation_count', 'high_voltage_substations', 'matched_substations',
            'estimated_daily_sessions', 'estimated_daily_demand_kwh', 'estimated_monthly_demand_kwh'
        ]
        
        for col in count_cols:
            if col in ward_data.columns:
                ward_data[col] = ward_data[col].fillna(0).astype(int)
        
        # Fill NaN for average metrics with median
        avg_cols = ['avg_speed_limit_kmh', 'avg_frc', 'avg_voltage_kv', 'land_use_feasibility_score']
        for col in avg_cols:
            if col in ward_data.columns:
                median_val = ward_data[col].median()
                ward_data[col] = ward_data[col].fillna(median_val)
        
        # Fill NaN for length with 0
        if 'total_road_length_km' in ward_data.columns:
            ward_data['total_road_length_km'] = ward_data['total_road_length_km'].fillna(0)
        
        # Fill dominant zone with 'UNKNOWN'
        if 'dominant_zone' in ward_data.columns:
            ward_data['dominant_zone'] = ward_data['dominant_zone'].fillna('UNKNOWN')
        
        log_section(logger, "AGGREGATION COMPLETE", char="*")
        log_data_summary(logger, ward_data, "Ward-Level Aggregated Data")
        
        # Summary statistics
        logger.info("\n" + "="*70)
        logger.info("WARD-LEVEL SUMMARY STATISTICS")
        logger.info("="*70)
        logger.info(f"Total wards: {len(ward_data)}")
        logger.info(f"Wards with traffic data: {(ward_data['total_traffic_segments'] > 0).sum()}")
        logger.info(f"Wards with existing chargers: {(ward_data['existing_chargers_count'] > 0).sum()}")
        logger.info(f"Wards with grid substations: {(ward_data['substation_count'] > 0).sum()}")
        logger.info(f"Average feasibility score: {ward_data['land_use_feasibility_score'].mean():.3f}")
        logger.info(f"Total road network: {ward_data['total_road_length_km'].sum():.1f} km")
        logger.info(f"Total existing chargers: {ward_data['existing_chargers_count'].sum()}")
        logger.info(f"Total daily demand: {ward_data['estimated_daily_demand_kwh'].sum():,} kWh")
        logger.info("="*70)
        
        return ward_data


if __name__ == "__main__":
    """Test aggregation module"""
    
    logger.info("Testing ward aggregation module...")
    
    # Import data processing modules
    from data_loader import DataLoader
    from spatial_join import SpatialJoiner
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Perform spatial joins
    joiner = SpatialJoiner(data['wards'])
    joined_data = joiner.join_all_datasets(data)
    
    # Aggregate to ward level
    aggregator = WardAggregator()
    ward_level_data = aggregator.aggregate_all(joined_data, data['wards'])
    
    # Save to CSV for inspection
    output_file = OUTPUT_DIR / 'ward_level_data.csv'
    ward_level_data.drop(columns=['geometry']).to_csv(output_file, index=False)
    logger.info(f"Saved ward-level data to {output_file}")
    
    # Save as GeoJSON for mapping
    geojson_file = OUTPUT_DIR / 'ward_level_data.geojson'
    ward_level_data.to_file(geojson_file, driver='GeoJSON')
    logger.info(f"Saved ward-level GeoJSON to {geojson_file}")
    
    logger.info("\n" + "="*70)
    logger.info("Aggregation Test Complete")
    logger.info("="*70)
    
    # Display sample records
    print("\nSample ward-level data (first 5 wards):")
    print(ward_level_data.drop(columns=['geometry']).head().to_string())
