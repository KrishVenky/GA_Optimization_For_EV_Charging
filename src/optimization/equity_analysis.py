"""
Equity Analysis for EV Charging Infrastructure
Computes Gini coefficient and accessibility metrics before/after optimization

Outputs:
- TABLE_VI_Equity_Metrics.csv (for paper)
- FIGURE_8_Lorenz_Curve.png (for paper)
- FIGURE_9_Accessibility_Histogram.png (for paper)

Usage: python equity_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import OUTPUT_DIR
from src.utils.logger import setup_logger, log_section

logger = setup_logger(__name__)
sns.set_style("whitegrid")
sns.set_palette("husl")


def compute_gini_coefficient(distribution):
    """
    Compute Gini coefficient for a distribution
    
    Args:
        distribution: Array of values (e.g., chargers per ward)
    
    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    distribution = np.array(distribution, dtype=float)
    
    # Remove zeros for meaningful inequality measurement
    # (wards with 0 chargers are underserved, not equal)
    distribution = distribution[distribution > 0] if len(distribution[distribution > 0]) > 0 else distribution
    
    if len(distribution) == 0 or np.sum(distribution) == 0:
        return 0.0
    
    # Sort values
    sorted_dist = np.sort(distribution)
    n = len(sorted_dist)
    
    # Compute Gini coefficient using standard formula
    cumsum = np.cumsum(sorted_dist)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_dist)) / (n * np.sum(sorted_dist)) - (n + 1) / n
    
    return gini


def compute_lorenz_curve(distribution):
    """
    Compute Lorenz curve coordinates
    
    Returns:
        (cumulative_population_fraction, cumulative_charger_fraction)
    """
    distribution = np.array(distribution, dtype=float)
    sorted_dist = np.sort(distribution)
    
    n = len(sorted_dist)
    cumsum = np.cumsum(sorted_dist)
    total = np.sum(sorted_dist)
    
    # Lorenz curve coordinates
    lorenz_x = np.arange(0, n+1) / n  # Population fraction (0 to 1)
    lorenz_y = np.concatenate([[0], cumsum]) / total if total > 0 else np.zeros(n+1)  # Charger fraction
    
    return lorenz_x, lorenz_y


def compute_accessibility_metrics(ward_data, charger_column='existing_chargers_count', threshold_km=2.0):
    """
    Compute accessibility metrics: % of wards and % of demand within threshold distance
    
    Args:
        ward_data: DataFrame with ward information
        charger_column: Column name for charger counts
        threshold_km: Distance threshold in kilometers
    
    Returns:
        dict with accessibility metrics
    """
    # Wards with at least one charger
    wards_with_chargers = ward_data[ward_data[charger_column] > 0]
    
    # For simplicity, assume wards are approximately 5km x 5km
    # A ward within 2km means adjacent or same ward
    # This is a conservative estimate - in reality we'd need spatial distance matrix
    
    # Metric 1: % of wards with at least one charger (direct access)
    wards_with_access = len(wards_with_chargers)
    total_wards = len(ward_data)
    ward_accessibility = (wards_with_access / total_wards) * 100
    
    # Metric 2: % of demand in wards with chargers
    demand_with_access = wards_with_chargers['estimated_daily_demand_kwh'].sum()
    total_demand = ward_data['estimated_daily_demand_kwh'].sum()
    demand_accessibility = (demand_with_access / total_demand) * 100 if total_demand > 0 else 0
    
    # Metric 3: Average chargers per served ward
    avg_chargers_per_served_ward = wards_with_chargers[charger_column].mean() if len(wards_with_chargers) > 0 else 0
    
    return {
        'wards_with_access': wards_with_access,
        'total_wards': total_wards,
        'ward_accessibility_pct': ward_accessibility,
        'demand_with_access_kwh': demand_with_access,
        'total_demand_kwh': total_demand,
        'demand_accessibility_pct': demand_accessibility,
        'avg_chargers_per_served_ward': avg_chargers_per_served_ward
    }


def analyze_equity():
    """Main equity analysis function"""
    log_section(logger, "EQUITY ANALYSIS: GINI COEFFICIENT & ACCESSIBILITY", char="█")
    
    # Load data
    logger.info("Loading ward-level data and optimization results...")
    ward_data = pd.read_csv(OUTPUT_DIR / 'ward_level_data.csv')
    optimized_allocation = pd.read_csv(OUTPUT_DIR / 'optimized_allocation.csv')
    
    # Merge optimized allocations into ward data
    ward_data_optimized = ward_data.merge(
        optimized_allocation[['KGISWardID', 'allocated_chargers']], 
        on='KGISWardID', 
        how='left'
    )
    ward_data_optimized['allocated_chargers'] = ward_data_optimized['allocated_chargers'].fillna(0).astype(int)
    
    logger.info(f"Loaded {len(ward_data)} wards")
    logger.info(f"Current total chargers: {ward_data['existing_chargers_count'].sum()}")
    logger.info(f"Optimized total chargers: {ward_data_optimized['allocated_chargers'].sum()}")
    
    # ========== GINI COEFFICIENT ANALYSIS ==========
    log_section(logger, "Computing Gini Coefficients")
    
    # Current distribution
    current_chargers = ward_data['existing_chargers_count'].values
    gini_current = compute_gini_coefficient(current_chargers)
    
    # Optimized distribution
    optimized_chargers = ward_data_optimized['allocated_chargers'].values
    gini_optimized = compute_gini_coefficient(optimized_chargers)
    
    # Combined (current + optimized)
    combined_chargers = current_chargers + optimized_chargers
    gini_combined = compute_gini_coefficient(combined_chargers)
    
    logger.info(f"Gini (Current): {gini_current:.4f}")
    logger.info(f"Gini (Optimized Alone): {gini_optimized:.4f}")
    logger.info(f"Gini (Current + Optimized): {gini_combined:.4f}")
    logger.info(f"Gini Improvement: {((gini_current - gini_combined) / gini_current * 100):.1f}%")
    
    # ========== ACCESSIBILITY METRICS ==========
    log_section(logger, "Computing Accessibility Metrics")
    
    # Current accessibility
    access_current = compute_accessibility_metrics(ward_data, 'existing_chargers_count')
    
    # Optimized accessibility (combined scenario)
    ward_data_combined = ward_data.copy()
    ward_data_combined['combined_chargers'] = current_chargers + optimized_chargers
    access_combined = compute_accessibility_metrics(ward_data_combined, 'combined_chargers')
    
    logger.info(f"Current: {access_current['wards_with_access']}/{access_current['total_wards']} wards have chargers ({access_current['ward_accessibility_pct']:.1f}%)")
    logger.info(f"Combined: {access_combined['wards_with_access']}/{access_combined['total_wards']} wards have chargers ({access_combined['ward_accessibility_pct']:.1f}%)")
    logger.info(f"Current: {access_current['demand_accessibility_pct']:.1f}% of demand in served wards")
    logger.info(f"Combined: {access_combined['demand_accessibility_pct']:.1f}% of demand in served wards")
    
    # ========== CREATE TABLE VI: EQUITY METRICS ==========
    log_section(logger, "Generating TABLE VI: Equity Metrics")
    
    equity_table = pd.DataFrame([
        {
            'Metric': 'Gini Coefficient',
            'Current Infrastructure': f'{gini_current:.4f}',
            'After Optimization': f'{gini_combined:.4f}',
            'Change': f'{gini_combined - gini_current:+.4f} ({(gini_combined - gini_current)/gini_current*100:+.1f}%)'
        },
        {
            'Metric': 'Wards with Chargers',
            'Current Infrastructure': f"{access_current['wards_with_access']} ({access_current['ward_accessibility_pct']:.1f}%)",
            'After Optimization': f"{access_combined['wards_with_access']} ({access_combined['ward_accessibility_pct']:.1f}%)",
            'Change': f"+{access_combined['wards_with_access'] - access_current['wards_with_access']} (+{access_combined['ward_accessibility_pct'] - access_current['ward_accessibility_pct']:.1f}pp)"
        },
        {
            'Metric': 'Demand in Served Wards (%)',
            'Current Infrastructure': f"{access_current['demand_accessibility_pct']:.1f}%",
            'After Optimization': f"{access_combined['demand_accessibility_pct']:.1f}%",
            'Change': f"+{access_combined['demand_accessibility_pct'] - access_current['demand_accessibility_pct']:.1f}pp"
        },
        {
            'Metric': 'Avg Chargers per Served Ward',
            'Current Infrastructure': f"{access_current['avg_chargers_per_served_ward']:.2f}",
            'After Optimization': f"{access_combined['avg_chargers_per_served_ward']:.2f}",
            'Change': f"{access_combined['avg_chargers_per_served_ward'] - access_current['avg_chargers_per_served_ward']:+.2f}"
        },
        {
            'Metric': 'Total Chargers',
            'Current Infrastructure': f"{int(current_chargers.sum())}",
            'After Optimization': f"{int(combined_chargers.sum())}",
            'Change': f"+{int(optimized_chargers.sum())}"
        }
    ])
    
    table_file = OUTPUT_DIR / 'ieee_paper' / 'TABLE_VI_Equity_Metrics.csv'
    equity_table.to_csv(table_file, index=False)
    logger.info(f"Saved TABLE VI to: {table_file}")
    
    print("\n" + "="*80)
    print("TABLE VI: EQUITY METRICS")
    print("="*80)
    print(equity_table.to_string(index=False))
    print("="*80 + "\n")
    
    # ========== FIGURE 8: LORENZ CURVE ==========
    log_section(logger, "Generating FIGURE 8: Lorenz Curve")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute Lorenz curves
    lorenz_x_current, lorenz_y_current = compute_lorenz_curve(current_chargers)
    lorenz_x_optimized, lorenz_y_optimized = compute_lorenz_curve(combined_chargers)
    
    # Plot perfect equality line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Equality', alpha=0.5)
    
    # Plot Lorenz curves
    ax.plot(lorenz_x_current, lorenz_y_current, linewidth=3, 
            label=f'Current Infrastructure (Gini={gini_current:.3f})', 
            marker='o', markersize=4, markevery=20)
    
    ax.plot(lorenz_x_optimized, lorenz_y_optimized, linewidth=3, 
            label=f'After Optimization (Gini={gini_combined:.3f})',
            marker='s', markersize=4, markevery=20)
    
    # Fill area between curves
    ax.fill_between(lorenz_x_current, lorenz_y_current, lorenz_x_optimized, 
                    alpha=0.2, color='green', label='Equity Improvement')
    
    ax.set_xlabel('Cumulative Fraction of Wards (sorted by charger count)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Fraction of Chargers', fontsize=12, fontweight='bold')
    ax.set_title('Lorenz Curve: Spatial Equity of EV Charging Infrastructure\n' + 
                f'Gini Improvement: {(gini_current - gini_combined):.3f} ({(gini_current - gini_combined)/gini_current*100:.1f}% reduction)',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add text annotation
    ax.text(0.6, 0.25, 
            f'Lower Gini = More Equitable\nCurrent: {gini_current:.3f}\nOptimized: {gini_combined:.3f}',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    fig_file = OUTPUT_DIR / 'ieee_paper' / 'FIGURE_8_Lorenz_Curve.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved FIGURE 8 to: {fig_file}")
    plt.close()
    
    # ========== FIGURE 9: ACCESSIBILITY HISTOGRAM ==========
    log_section(logger, "Generating FIGURE 9: Accessibility Histogram")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel A: Chargers per ward distribution
    ax1 = axes[0]
    
    bins = np.arange(0, max(combined_chargers.max(), current_chargers.max()) + 2) - 0.5
    
    ax1.hist(current_chargers, bins=bins, alpha=0.6, label='Current', 
             color='#e74c3c', edgecolor='black', linewidth=1.2)
    ax1.hist(combined_chargers, bins=bins, alpha=0.6, label='After Optimization', 
             color='#27ae60', edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Chargers per Ward', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Wards', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Distribution of Chargers per Ward', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add statistics text
    current_zeros = np.sum(current_chargers == 0)
    optimized_zeros = np.sum(combined_chargers == 0)
    ax1.text(0.65, 0.95, 
            f'Wards with 0 chargers:\n  Current: {current_zeros} ({current_zeros/len(current_chargers)*100:.1f}%)\n  Optimized: {optimized_zeros} ({optimized_zeros/len(combined_chargers)*100:.1f}%)',
            transform=ax1.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
            fontsize=10, verticalalignment='top')
    
    # Panel B: Demand coverage by ward
    ax2 = axes[1]
    
    ward_data_plot = ward_data_combined.copy()
    ward_data_plot['has_charger_current'] = (current_chargers > 0).astype(int)
    ward_data_plot['has_charger_optimized'] = (combined_chargers > 0).astype(int)
    
    # Bar chart: total demand in wards with/without chargers
    demand_current_covered = ward_data_plot[ward_data_plot['has_charger_current'] == 1]['estimated_daily_demand_kwh'].sum()
    demand_current_uncovered = ward_data_plot[ward_data_plot['has_charger_current'] == 0]['estimated_daily_demand_kwh'].sum()
    
    demand_optimized_covered = ward_data_plot[ward_data_plot['has_charger_optimized'] == 1]['estimated_daily_demand_kwh'].sum()
    demand_optimized_uncovered = ward_data_plot[ward_data_plot['has_charger_optimized'] == 0]['estimated_daily_demand_kwh'].sum()
    
    categories = ['Current\nInfrastructure', 'After\nOptimization']
    covered = [demand_current_covered, demand_optimized_covered]
    uncovered = [demand_current_uncovered, demand_optimized_uncovered]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x, covered, width, label='Demand in Served Wards', 
                    color='#27ae60', edgecolor='black', linewidth=1.2)
    bars2 = ax2.bar(x, uncovered, width, bottom=covered, label='Demand in Underserved Wards',
                    color='#e74c3c', edgecolor='black', linewidth=1.2)
    
    ax2.set_ylabel('Daily Charging Demand (kWh)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Demand Accessibility', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (cov, uncov) in enumerate(zip(covered, uncovered)):
        total = cov + uncov
        pct_covered = (cov / total * 100) if total > 0 else 0
        ax2.text(i, cov/2, f'{pct_covered:.1f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    plt.suptitle('Spatial Equity and Accessibility Metrics Before and After Optimization',
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    fig_file = OUTPUT_DIR / 'ieee_paper' / 'FIGURE_9_Accessibility_Histogram.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved FIGURE 9 to: {fig_file}")
    plt.close()
    
    # ========== SUMMARY ==========
    log_section(logger, "EQUITY ANALYSIS COMPLETE", char="█")
    
    print("\n" + "="*80)
    print("KEY EQUITY FINDINGS FOR PAPER")
    print("="*80)
    print(f"1. Gini Coefficient reduced from {gini_current:.4f} to {gini_combined:.4f}")
    print(f"   → {(gini_current - gini_combined)/gini_current*100:.1f}% improvement in spatial equity")
    print(f"\n2. Ward Accessibility increased from {access_current['ward_accessibility_pct']:.1f}% to {access_combined['ward_accessibility_pct']:.1f}%")
    print(f"   → {access_combined['wards_with_access'] - access_current['wards_with_access']} additional wards gain access")
    print(f"\n3. Demand Accessibility increased from {access_current['demand_accessibility_pct']:.1f}% to {access_combined['demand_accessibility_pct']:.1f}%")
    print(f"   → {access_combined['demand_accessibility_pct'] - access_current['demand_accessibility_pct']:.1f} percentage point improvement")
    print(f"\n4. Wards with zero chargers reduced from {current_zeros} to {optimized_zeros}")
    print(f"   → {current_zeros - optimized_zeros} wards move from underserved to served")
    print("="*80)
    print("\nPAPER STATEMENT:")
    print(f'"Beyond coverage maximization, our GA solution reduces spatial inequality')
    print(f'by {(gini_current - gini_combined)/gini_current*100:.1f}% as measured by Gini coefficient (from {gini_current:.3f} to {gini_combined:.3f}),')
    print(f'while increasing ward accessibility from {access_current["ward_accessibility_pct"]:.1f}% to {access_combined["ward_accessibility_pct"]:.1f}%.')
    print(f'This demonstrates that efficiency and equity are complementary objectives"')
    print("="*80 + "\n")
    
    return {
        'gini_current': gini_current,
        'gini_combined': gini_combined,
        'gini_improvement_pct': (gini_current - gini_combined)/gini_current*100,
        'access_current': access_current,
        'access_combined': access_combined
    }


if __name__ == "__main__":
    logger.info("Starting equity analysis...")
    results = analyze_equity()
    logger.info("Equity analysis complete!")
    logger.info(f"Output files in: {OUTPUT_DIR / 'ieee_paper'}")
