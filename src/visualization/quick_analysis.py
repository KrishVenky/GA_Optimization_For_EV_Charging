"""
Quick exploratory analysis and visualization

Generates summary statistics and identifies priority wards for EV charging infrastructure
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import OUTPUT_DIR
from src.utils.logger import setup_logger, log_section

logger = setup_logger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_ward_data():
    """Load ward-level aggregated data"""
    csv_file = OUTPUT_DIR / 'ward_level_data.csv'
    geojson_file = OUTPUT_DIR / 'ward_level_data.geojson'
    
    if geojson_file.exists():
        logger.info(f"Loading ward data from {geojson_file}")
        return gpd.read_file(geojson_file)
    elif csv_file.exists():
        logger.info(f"Loading ward data from {csv_file}")
        return pd.read_csv(csv_file)
    else:
        raise FileNotFoundError("Ward-level data not found. Run aggregation.py first.")


def identify_priority_wards(ward_data: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identify priority wards for new charging infrastructure
    
    Priority scoring criteria:
    - High estimated demand (traffic-based)
    - Low existing charger density
    - High land use feasibility
    - Good grid infrastructure (where available)
    
    Args:
        ward_data: Ward-level data
        top_n: Number of top priority wards to return
    
    Returns:
        DataFrame with priority wards sorted by score
    """
    log_section(logger, "IDENTIFYING PRIORITY WARDS")
    
    # Calculate normalized scores (0-1 range)
    df = ward_data.copy()
    
    # Demand score (higher is better)
    demand_max = df['estimated_daily_demand_kwh'].max()
    df['demand_score'] = df['estimated_daily_demand_kwh'] / demand_max if demand_max > 0 else 0
    
    # Gap score: inverse of existing charger density (higher gap = higher priority)
    # Wards with 0 chargers get max score
    charger_max = df['existing_chargers_count'].max()
    if charger_max > 0:
        df['gap_score'] = 1 - (df['existing_chargers_count'] / charger_max)
    else:
        df['gap_score'] = 1.0
    
    # Feasibility score (already 0-1)
    df['feasibility_score'] = df['land_use_feasibility_score']
    
    # Grid score (wards with substations get bonus)
    df['grid_score'] = (df['substation_count'] > 0).astype(float)
    
    # Composite priority score (weighted average)
    weights = {
        'demand': 0.40,      # 40% - Demand is primary driver
        'gap': 0.35,         # 35% - Filling gaps is critical
        'feasibility': 0.20, # 20% - Land use feasibility
        'grid': 0.05         # 5% - Grid proximity (bonus)
    }
    
    df['priority_score'] = (
        df['demand_score'] * weights['demand'] +
        df['gap_score'] * weights['gap'] +
        df['feasibility_score'] * weights['feasibility'] +
        df['grid_score'] * weights['grid']
    )
    
    # Sort by priority
    priority_wards = df.nlargest(top_n, 'priority_score')
    
    logger.info(f"Top {top_n} priority wards identified")
    logger.info(f"Scoring weights: {weights}")
    
    return priority_wards[[
        'KGISWardID', 'KGISWardName', 'priority_score',
        'estimated_daily_demand_kwh', 'existing_chargers_count',
        'land_use_feasibility_score', 'substation_count',
        'total_road_length_km', 'dominant_zone'
    ]]


def generate_summary_statistics(ward_data: pd.DataFrame):
    """Generate comprehensive summary statistics"""
    log_section(logger, "SUMMARY STATISTICS", char="=")
    
    print("\n" + "="*80)
    print("BENGALURU URBAN EV CHARGING INFRASTRUCTURE ANALYSIS")
    print("="*80)
    
    # Overall statistics
    print("\n### CITYWIDE METRICS ###")
    print(f"Total wards analyzed: {len(ward_data)}")
    print(f"Total road network: {ward_data['total_road_length_km'].sum():,.1f} km")
    print(f"Average road per ward: {ward_data['total_road_length_km'].mean():,.1f} km")
    print(f"Total existing chargers: {ward_data['existing_chargers_count'].sum():.0f}")
    print(f"Total charging points: {ward_data['total_charging_points'].sum():.0f}")
    print(f"Estimated daily demand: {ward_data['estimated_daily_demand_kwh'].sum():,.0f} kWh")
    print(f"Estimated monthly demand: {ward_data['estimated_monthly_demand_kwh'].sum():,.0f} kWh")
    
    # Infrastructure gaps
    print("\n### INFRASTRUCTURE GAPS ###")
    wards_no_chargers = (ward_data['existing_chargers_count'] == 0).sum()
    wards_no_traffic = (ward_data['total_traffic_segments'] == 0).sum()
    wards_no_grid = (ward_data['substation_count'] == 0).sum()
    
    print(f"Wards with NO chargers: {wards_no_chargers} ({wards_no_chargers/len(ward_data)*100:.1f}%)")
    print(f"Wards with NO traffic data: {wards_no_traffic} ({wards_no_traffic/len(ward_data)*100:.1f}%)")
    print(f"Wards with NO substations: {wards_no_grid} ({wards_no_grid/len(ward_data)*100:.1f}%)")
    
    # Concentration analysis
    print("\n### CONCENTRATION ANALYSIS ###")
    # Top 10% of wards by chargers
    top_10pct = int(len(ward_data) * 0.1)
    top_wards_chargers = ward_data.nlargest(top_10pct, 'existing_chargers_count')
    charger_concentration = top_wards_chargers['existing_chargers_count'].sum() / ward_data['existing_chargers_count'].sum() * 100
    
    print(f"Top 10% of wards have {charger_concentration:.1f}% of all chargers")
    print(f"Charger Gini coefficient: {calculate_gini(ward_data['existing_chargers_count']):.3f} (0=equal, 1=concentrated)")
    
    # Demand vs Supply mismatch
    print("\n### DEMAND-SUPPLY ANALYSIS ###")
    # Wards with high demand but no chargers
    high_demand_no_chargers = ward_data[
        (ward_data['estimated_daily_demand_kwh'] > ward_data['estimated_daily_demand_kwh'].median()) &
        (ward_data['existing_chargers_count'] == 0)
    ]
    print(f"High-demand wards with NO chargers: {len(high_demand_no_chargers)}")
    
    # Average feasibility
    print(f"\nAverage land use feasibility: {ward_data['land_use_feasibility_score'].mean():.3f}")
    print(f"Wards with high feasibility (>0.75): {(ward_data['land_use_feasibility_score'] > 0.75).sum()}")
    
    # Zone distribution
    print("\n### ZONE DISTRIBUTION ###")
    zone_counts = ward_data['dominant_zone'].value_counts()
    for zone, count in zone_counts.items():
        print(f"{zone:20s}: {count:3d} wards ({count/len(ward_data)*100:5.1f}%)")
    
    print("="*80 + "\n")


def calculate_gini(values):
    """Calculate Gini coefficient (inequality measure)"""
    import numpy as np
    values = values.fillna(0).values
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = 0
    total = sorted_values.sum()
    if total == 0:
        return 0
    for i, val in enumerate(sorted_values):
        cumsum += (n - i) * val
    return (n + 1 - 2 * cumsum / total) / n


def create_visualizations(ward_data: pd.DataFrame, output_dir: Path):
    """Create basic visualizations"""
    log_section(logger, "GENERATING VISUALIZATIONS")
    
    # 1. Distribution of existing chargers
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Charger distribution
    ax1 = axes[0, 0]
    charger_counts = ward_data['existing_chargers_count'].value_counts().sort_index()
    ax1.bar(charger_counts.index, charger_counts.values, color='steelblue')
    ax1.set_xlabel('Number of Chargers per Ward')
    ax1.set_ylabel('Number of Wards')
    ax1.set_title('Distribution of Existing Chargers Across Wards')
    ax1.grid(True, alpha=0.3)
    
    # Demand distribution
    ax2 = axes[0, 1]
    ax2.hist(ward_data['estimated_daily_demand_kwh'], bins=30, color='orange', edgecolor='black')
    ax2.set_xlabel('Estimated Daily Demand (kWh)')
    ax2.set_ylabel('Number of Wards')
    ax2.set_title('Distribution of Estimated Charging Demand')
    ax2.grid(True, alpha=0.3)
    
    # Feasibility by zone
    ax3 = axes[1, 0]
    zone_feasibility = ward_data.groupby('dominant_zone')['land_use_feasibility_score'].mean().sort_values()
    zone_feasibility.plot(kind='barh', ax=ax3, color='green')
    ax3.set_xlabel('Average Feasibility Score')
    ax3.set_title('Land Use Feasibility by Dominant Zone')
    ax3.grid(True, alpha=0.3)
    
    # Demand vs Chargers scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        ward_data['estimated_daily_demand_kwh'],
        ward_data['existing_chargers_count'],
        c=ward_data['land_use_feasibility_score'],
        cmap='RdYlGn',
        alpha=0.6,
        s=50
    )
    ax4.set_xlabel('Estimated Daily Demand (kWh)')
    ax4.set_ylabel('Existing Chargers Count')
    ax4.set_title('Demand vs Supply (color = feasibility)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Feasibility')
    
    plt.tight_layout()
    output_file = output_dir / 'analysis_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {output_file}")
    plt.close()


if __name__ == "__main__":
    logger.info("Starting quick analysis...")
    
    # Load data
    ward_data = load_ward_data()
    logger.info(f"Loaded {len(ward_data)} wards")
    
    # Generate statistics
    generate_summary_statistics(ward_data)
    
    # Identify priority wards
    priority_wards = identify_priority_wards(ward_data, top_n=20)
    
    print("\n" + "="*80)
    print("TOP 20 PRIORITY WARDS FOR NEW CHARGING INFRASTRUCTURE")
    print("="*80)
    print(priority_wards.to_string(index=False))
    print("="*80 + "\n")
    
    # Save priority wards
    priority_output = OUTPUT_DIR / 'priority_wards.csv'
    priority_wards.to_csv(priority_output, index=False)
    logger.info(f"Saved priority wards to {priority_output}")
    
    # Create visualizations
    create_visualizations(ward_data, OUTPUT_DIR)
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Outputs saved to {OUTPUT_DIR}")
