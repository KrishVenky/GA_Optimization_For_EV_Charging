"""
Spatial join module for EV charging infrastructure optimization
Performs coordinate-based matching to assign all features to BBMP wards
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Tuple, Dict
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *
from utils.logger import setup_logger, log_section, log_data_summary

logger = setup_logger(__name__)


class SpatialJoiner:
    """Handles spatial joins to assign all features to BBMP wards"""
    
    def __init__(self, wards: gpd.GeoDataFrame):
        """
        Initialize SpatialJoiner with ward boundaries
        
        Args:
            wards: GeoDataFrame with BBMP ward polygons
        """
        self.wards = wards
        logger.info(f"SpatialJoiner initialized with {len(wards)} wards")
        
        # Define projected CRS for Bangalore (UTM Zone 43N)
        self.projected_crs = 'EPSG:32643'  # WGS84 / UTM zone 43N (covers Bangalore)
        self.geographic_crs = 'EPSG:4326'  # WGS84 lat/lon
        
        logger.info(f"Using {self.projected_crs} for geometric operations")
        
        # Ensure wards have a unique identifier
        if 'KGISWardID' in wards.columns:
            self.ward_id_col = 'KGISWardID'
        elif 'KGISWardNo' in wards.columns:
            self.ward_id_col = 'KGISWardNo'
        else:
            logger.warning("No standard ward ID found, creating index-based ID")
            self.wards['ward_id'] = range(len(wards))
            self.ward_id_col = 'ward_id'
        
        logger.info(f"Using '{self.ward_id_col}' as ward identifier")
    
    def join_traffic_to_wards(self, traffic: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign traffic segments to wards using centroid method
        
        Args:
            traffic: GeoDataFrame with traffic LineStrings
        
        Returns:
            Traffic data with ward assignments
        """
        log_section(logger, "Joining Traffic Segments to Wards")
        
        logger.info(f"Processing {len(traffic)} traffic segments")
        
        # Create centroid geometry for spatial join
        logger.info("Computing centroids of traffic segments...")
        traffic_points = traffic.copy()
        
        # Reproject to UTM for accurate centroid calculation
        logger.info(f"Reprojecting to {self.projected_crs} for geometric operations...")
        traffic_points = traffic_points.to_crs(self.projected_crs)
        traffic_points['geometry'] = traffic_points['geometry'].centroid
        traffic_points = traffic_points.to_crs(self.geographic_crs)  # Back to WGS84
        
        # Perform spatial join
        logger.info("Performing spatial join (point-in-polygon)...")
        joined = gpd.sjoin(
            traffic_points,
            self.wards[[self.ward_id_col, 'KGISWardName', 'geometry']],
            how='left',
            predicate='within'
        )
        
        # Restore original geometry
        joined['geometry'] = traffic['geometry'].values
        
        # Calculate coverage
        matched = joined[self.ward_id_col].notna().sum()
        coverage = (matched / len(joined)) * 100
        logger.info(f"Matched {matched}/{len(joined)} segments ({coverage:.1f}% coverage)")
        
        # Report unmatched segments
        unmatched = len(joined) - matched
        if unmatched > 0:
            logger.warning(f"{unmatched} segments did not match any ward (outside boundaries)")
        
        log_data_summary(logger, joined, "Traffic with Ward Assignments")
        return joined
    
    def join_chargers_to_wards(self, chargers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign existing chargers to wards using point-in-polygon
        
        Args:
            chargers: GeoDataFrame with charger Points
        
        Returns:
            Charger data with ward assignments
        """
        log_section(logger, "Joining Existing Chargers to Wards")
        
        logger.info(f"Processing {len(chargers)} existing chargers")
        
        # Perform spatial join
        joined = gpd.sjoin(
            chargers,
            self.wards[[self.ward_id_col, 'KGISWardName', 'geometry']],
            how='left',
            predicate='within'
        )
        
        # Calculate coverage
        matched = joined[self.ward_id_col].notna().sum()
        coverage = (matched / len(joined)) * 100
        logger.info(f"Matched {matched}/{len(joined)} chargers ({coverage:.1f}% coverage)")
        
        log_data_summary(logger, joined, "Chargers with Ward Assignments")
        return joined
    
    def join_landuse_to_wards(self, landuse: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Calculate land use overlap with wards (polygon intersection)
        
        Args:
            landuse: GeoDataFrame with land use zone polygons
        
        Returns:
            DataFrame with ward-level land use area percentages
        """
        log_section(logger, "Joining Land Use Zones to Wards")
        
        logger.info(f"Processing {len(landuse)} land use zones")
        
        # Perform overlay to get intersections
        logger.info("Computing polygon intersections...")
        
        # Reproject to UTM for accurate area calculations
        logger.info(f"Reprojecting to {self.projected_crs} for area calculations...")
        wards_proj = self.wards[[self.ward_id_col, 'KGISWardName', 'geometry']].to_crs(self.projected_crs)
        landuse_proj = landuse[['ZoneName', 'ZONEID', 'geometry']].to_crs(self.projected_crs)
        
        overlay = gpd.overlay(
            wards_proj,
            landuse_proj,
            how='intersection'
        )
        
        # Calculate intersection areas (in square meters)
        overlay['intersection_area_sqm'] = overlay.geometry.area
        
        # Calculate ward total areas (in square meters)
        ward_areas = wards_proj.copy()
        ward_areas['ward_area_sqm'] = ward_areas.geometry.area
        
        # Merge to get percentages
        overlay = overlay.merge(
            ward_areas[[self.ward_id_col, 'ward_area_sqm']],
            on=self.ward_id_col,
            how='left'
        )
        
        overlay['landuse_pct'] = (overlay['intersection_area_sqm'] / overlay['ward_area_sqm']) * 100
        
        logger.info(f"Created {len(overlay)} ward-landuse intersection records")
        logger.info(f"Zones covered: {overlay['ZoneName'].nunique()} unique zones")
        
        # Pivot to get one row per ward with zone percentages
        ward_landuse = overlay.pivot_table(
            index=self.ward_id_col,
            columns='ZoneName',
            values='landuse_pct',
            aggfunc='sum',
            fill_value=0
        ).reset_index()
        
        # Add ward names
        ward_landuse = ward_landuse.merge(
            self.wards[[self.ward_id_col, 'KGISWardName']],
            on=self.ward_id_col,
            how='left'
        )
        
        logger.info(f"Final ward-level land use data: {len(ward_landuse)} wards")
        log_data_summary(logger, ward_landuse, "Ward-Level Land Use")
        return ward_landuse
    
    def geocode_and_join_grid(self, grid: pd.DataFrame) -> pd.DataFrame:
        """
        Geocode grid substations and assign to wards
        
        Note: This is a simplified version. In production, you'd use a geocoding service.
        For now, we'll do a fuzzy match on Taluk names to approximate location.
        
        Args:
            grid: DataFrame with substation data (no coordinates)
        
        Returns:
            Grid data with ward assignments
        """
        log_section(logger, "Geocoding and Joining Grid Substations to Wards")
        
        logger.info(f"Processing {len(grid)} substations")
        logger.warning("Using Taluk-based fuzzy matching (not precise geocoding)")
        
        # Extract unique taluks from wards (from names or separate data)
        # This is a simplified approach - in reality, you'd geocode the full address
        
        # For now, we'll use a simple string matching approach
        # Assuming some ward names might contain locality info
        grid_with_wards = grid.copy()
        grid_with_wards[self.ward_id_col] = None
        grid_with_wards['KGISWardName'] = None
        grid_with_wards['match_method'] = 'unmatched'
        
        # Try to match substations to wards by name similarity
        for idx, substation in grid.iterrows():
            ss_name = str(substation['Name of Sub-Station']).lower()
            ss_taluk = str(substation['Taluk']).lower()
            
            # Check if any ward name contains the substation name or taluk
            for ward_idx, ward in self.wards.iterrows():
                ward_name = str(ward.get('KGISWardName', '')).lower()
                
                # Simple fuzzy match
                if ss_name in ward_name or ss_taluk in ward_name:
                    grid_with_wards.at[idx, self.ward_id_col] = ward[self.ward_id_col]
                    grid_with_wards.at[idx, 'KGISWardName'] = ward['KGISWardName']
                    grid_with_wards.at[idx, 'match_method'] = 'name_match'
                    break
        
        # Calculate coverage
        matched = grid_with_wards[self.ward_id_col].notna().sum()
        coverage = (matched / len(grid_with_wards)) * 100
        logger.warning(f"Matched {matched}/{len(grid_with_wards)} substations ({coverage:.1f}% coverage)")
        logger.warning("Low coverage expected - proper geocoding API needed for production")
        
        # For substations without ward match, we'll distribute them across all wards
        # based on Taluk boundaries (simplified assumption)
        unmatched = grid_with_wards[grid_with_wards[self.ward_id_col].isna()]
        if len(unmatched) > 0:
            logger.info(f"Distributing {len(unmatched)} unmatched substations across all wards")
            grid_with_wards.loc[grid_with_wards[self.ward_id_col].isna(), 'match_method'] = 'distributed'
        
        log_data_summary(logger, grid_with_wards, "Grid with Ward Assignments")
        return grid_with_wards
    
    def generate_coverage_report(self, joined_data: Dict) -> pd.DataFrame:
        """
        Generate coverage statistics report
        
        Args:
            joined_data: Dictionary with all spatially joined datasets
        
        Returns:
            DataFrame with coverage statistics
        """
        log_section(logger, "Generating Coverage Report")
        
        report = []
        
        # Traffic coverage
        traffic = joined_data['traffic']
        traffic_matched = traffic[self.ward_id_col].notna().sum()
        traffic_total = len(traffic)
        report.append({
            'dataset': 'Traffic Segments',
            'total_features': traffic_total,
            'matched_features': traffic_matched,
            'coverage_pct': (traffic_matched / traffic_total) * 100,
            'wards_covered': traffic[traffic[self.ward_id_col].notna()][self.ward_id_col].nunique()
        })
        
        # Chargers coverage
        chargers = joined_data['chargers']
        chargers_matched = chargers[self.ward_id_col].notna().sum()
        chargers_total = len(chargers)
        report.append({
            'dataset': 'Existing Chargers',
            'total_features': chargers_total,
            'matched_features': chargers_matched,
            'coverage_pct': (chargers_matched / chargers_total) * 100,
            'wards_covered': chargers[chargers[self.ward_id_col].notna()][self.ward_id_col].nunique()
        })
        
        # Grid coverage
        grid = joined_data['grid']
        grid_matched = grid[grid['match_method'] == 'name_match'].shape[0]
        grid_total = len(grid)
        report.append({
            'dataset': 'Grid Substations',
            'total_features': grid_total,
            'matched_features': grid_matched,
            'coverage_pct': (grid_matched / grid_total) * 100,
            'wards_covered': grid[grid[self.ward_id_col].notna()][self.ward_id_col].nunique()
        })
        
        # Land use coverage (all wards covered)
        landuse = joined_data['landuse']
        report.append({
            'dataset': 'Land Use Zones',
            'total_features': len(self.wards),
            'matched_features': len(landuse),
            'coverage_pct': 100.0,
            'wards_covered': len(landuse)
        })
        
        report_df = pd.DataFrame(report)
        
        logger.info("\n" + "="*70)
        logger.info("SPATIAL JOIN COVERAGE REPORT")
        logger.info("="*70)
        for _, row in report_df.iterrows():
            logger.info(f"{row['dataset']:25s}: {row['matched_features']:5.0f}/{row['total_features']:5.0f} ({row['coverage_pct']:5.1f}%) - {row['wards_covered']:3.0f} wards")
        logger.info("="*70)
        
        # Flag low-coverage datasets
        low_coverage = report_df[report_df['coverage_pct'] < 70]
        if not low_coverage.empty:
            logger.warning("DATASETS WITH LOW COVERAGE (<70%):")
            for _, row in low_coverage.iterrows():
                logger.warning(f"   - {row['dataset']}: {row['coverage_pct']:.1f}% coverage")
                if row['dataset'] == 'Grid Substations':
                    logger.warning("     -> Recommend: Use proper geocoding API for substation addresses")
        
        return report_df
    
    def join_all_datasets(self, data: Dict) -> Dict:
        """
        Perform all spatial joins
        
        Args:
            data: Dictionary with all loaded datasets from DataLoader
        
        Returns:
            Dictionary with spatially joined datasets
        """
        log_section(logger, "PERFORMING ALL SPATIAL JOINS", char="*")
        
        joined_data = {
            'wards': self.wards,
            'traffic': self.join_traffic_to_wards(data['traffic']),
            'chargers': self.join_chargers_to_wards(data['chargers']),
            'landuse': self.join_landuse_to_wards(data['landuse']),
            'grid': self.geocode_and_join_grid(data['grid']),
            'costs': data['costs'],  # No spatial join needed
            'fleet': data['fleet']    # No spatial join needed
        }
        
        # Generate coverage report
        coverage_report = self.generate_coverage_report(joined_data)
        joined_data['coverage_report'] = coverage_report
        
        log_section(logger, "SPATIAL JOINS COMPLETE", char="*")
        return joined_data


if __name__ == "__main__":
    # Test spatial joins
    from data_loader import DataLoader
    
    logger.info("Testing spatial join module...")
    
    # Load data
    loader = DataLoader()
    data = loader.load_all_data()
    
    # Perform spatial joins
    joiner = SpatialJoiner(data['wards'])
    joined_data = joiner.join_all_datasets(data)
    
    print("\n" + "="*60)
    print("Spatial Join Test Complete")
    print("="*60)
    print(f"\nTraffic segments with wards: {len(joined_data['traffic'])}")
    print(f"Chargers with wards: {len(joined_data['chargers'])}")
    print(f"Ward-level land use records: {len(joined_data['landuse'])}")
    print(f"Grid substations processed: {len(joined_data['grid'])}")
