"""
Configuration file for EV Charging Infrastructure Optimization Project
Contains all file paths, parameters, and settings
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==================== DATA FILE PATHS ====================

# Map/Boundary Data
BBMP_WARDS_FILE = DATA_DIR / "map" / "BBMP.geojson"

# Land Use Data
LANDUSE_ZONES_FILE = DATA_DIR / "landuse" / "ZONE_MapToKML.geojson"

# Traffic Data (all files in traffic directory)
TRAFFIC_DIR = DATA_DIR / "traffic"

# Grid Infrastructure
GRID_SUBSTATIONS_FILE = DATA_DIR / "grid" / "9f945eb2-b531-4d5f-a096-b9d6ddd7b1d4.csv"

# Existing Chargers
OPENCHARGEMAP_FILE = DATA_DIR / "chargers" / "openchargemap-bengaluru-20251111.geojson"
BESCOM_EV_STATS_FILE = DATA_DIR / "chargers" / "bescom-ev-2020-23.csv"

# Cost Data
TARIFFS_FILE = DATA_DIR / "cost" / "bescom_tariffs.csv"
INSTALLATION_COSTS_FILE = DATA_DIR / "cost" / "charger_installation.csv"
OPERATIONAL_COSTS_FILE = DATA_DIR / "cost" / "operational_costs.csv"
LAND_RENTAL_FILE = DATA_DIR / "cost" / "land_rental_rates.csv"

# Fleet Data
EV_FLEET_FILE = DATA_DIR / "fleet" / "EVIndia.csv"

# ==================== SPATIAL SETTINGS ====================

# Target Coordinate Reference System
TARGET_CRS = "EPSG:4326"  # WGS84 (lat/lon)

# Spatial join predicates
SPATIAL_JOIN_PREDICATE = "within"  # For point-in-polygon joins
SPATIAL_JOIN_METHOD = "centroid"  # For linestring joins: 'centroid', 'start', 'end'

# ==================== DATA PROCESSING SETTINGS ====================

# Traffic data aggregation
TRAFFIC_METRICS = [
    "speedLimit",
    "frc",  # Functional Road Class
    "distance",
    "probeCount"
]

# Grid capacity settings
VOLTAGE_CAPACITY_WEIGHTS = {
    400: 1.0,  # 400kV substations (highest capacity)
    220: 0.6,  # 220kV substations
    110: 0.3,  # 110kV substations (if present)
}

# Charger type categories
CHARGER_TYPES = ["AC001", "DC001", "DC 50kW", "DC150kW"]

# ==================== OPTIMIZATION SETTINGS ====================

# Genetic Algorithm Parameters
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_MUTATION_RATE = 0.15
GA_CROSSOVER_RATE = 0.8
GA_ELITE_SIZE = 10  # Number of top solutions to preserve

# Fitness function weights (must sum to 1.0)
WEIGHT_COST = 0.4  # Minimize total cost
WEIGHT_COVERAGE = 0.35  # Maximize demand coverage
WEIGHT_RELIABILITY = 0.25  # Maximize grid reliability/redundancy

# Constraint thresholds
MAX_CHARGERS_PER_WARD = 10  # Maximum number of charging stations per ward
MIN_GRID_CAPACITY_THRESHOLD = 0.2  # Minimum available grid capacity (20%)
BUDGET_CONSTRAINT = None  # Set to float value if budget limit exists (in ₹)

# ==================== COST MODEL SETTINGS ====================

# Annual projection period
PROJECTION_YEARS = 10

# Electricity usage assumptions (kWh per charger per day)
CHARGER_DAILY_USAGE = {
    "AC001": 150,
    "DC001": 300,
    "DC 50kW": 500,
    "DC150kW": 1000
}

# Utilization rate (percentage of time charger is in use)
CHARGER_UTILIZATION_RATE = 0.65  # 65%

# ==================== DEMAND ESTIMATION SETTINGS ====================

# EV adoption growth rate (annual)
EV_ADOPTION_GROWTH_RATE = 0.25  # 25% year-over-year

# Demand scaling factors
TRAFFIC_TO_DEMAND_MULTIPLIER = 0.05  # 5% of traffic volume converts to EV charging demand

# ==================== VISUALIZATION SETTINGS ====================

# Color schemes for heatmaps
DEMAND_COLORMAP = "YlOrRd"  # Yellow-Orange-Red
COVERAGE_COLORMAP = "Greens"
COST_COLORMAP = "Blues"

# Map settings
MAP_CENTER = [12.9716, 77.5946]  # Bangalore center (lat, lon)
MAP_ZOOM_START = 11

# ==================== LOGGING SETTINGS ====================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "optimization.log"

# ==================== VALIDATION SETTINGS ====================

# Data quality thresholds
MIN_SPATIAL_JOIN_COVERAGE = 0.70  # Warn if <70% of features are matched to wards
MAX_NULL_PERCENTAGE = 0.30  # Warn if >30% of ward data is null

# ==================== OUTPUT SETTINGS ====================

# Output file names
AGGREGATED_WARD_DATA = OUTPUT_DIR / "ward_level_data.csv"
OPTIMIZATION_RESULTS = OUTPUT_DIR / "optimal_charger_placement.csv"
OPTIMIZATION_RESULTS_GEOJSON = OUTPUT_DIR / "optimal_charger_placement.geojson"
SUMMARY_REPORT = OUTPUT_DIR / "optimization_report.txt"
CONVERGENCE_PLOT = OUTPUT_DIR / "ga_convergence.png"
COVERAGE_MAP = OUTPUT_DIR / "coverage_map.html"
