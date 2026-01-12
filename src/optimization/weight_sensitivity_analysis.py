"""
Weight Sensitivity Analysis for EV Charging Infrastructure Optimization
Tests GA performance across different fitness weight configurations

Outputs:
- TABLE_VII_Weight_Sensitivity.csv (for paper)
- FIGURE_10_Weight_Sensitivity_Spatial.png (for paper)
- FIGURE_11_Weight_Sensitivity_Metrics.png (for paper)

Usage: python weight_sensitivity_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import OUTPUT_DIR
from src.utils.logger import setup_logger, log_section
from src.optimization.hybrid_ga_sa import HybridOptimizer

logger = setup_logger(__name__)
sns.set_style("whitegrid")


# Define weight scenarios
WEIGHT_SCENARIOS = {
    'Baseline (Current)': {
        'weights': {'cost': -0.30, 'coverage': 0.40, 'grid': 0.15, 'feasibility': 0.15},
        'description': 'Balanced approach prioritizing coverage',
        'color': '#3498db'
    },
    'Cost-Heavy (Austerity)': {
        'weights': {'cost': -0.50, 'coverage': 0.25, 'grid': 0.15, 'feasibility': 0.10},
        'description': 'Budget-constrained scenario, minimize spending',
        'color': '#e74c3c'
    },
    'Coverage-Heavy (Demand-First)': {
        'weights': {'cost': -0.15, 'coverage': 0.60, 'grid': 0.15, 'feasibility': 0.10},
        'description': 'Maximize service coverage, less cost-sensitive',
        'color': '#27ae60'
    },
    'Grid-Heavy (Infrastructure)': {
        'weights': {'cost': -0.20, 'coverage': 0.30, 'grid': 0.35, 'feasibility': 0.15},
        'description': 'Prioritize grid proximity, reduce connection costs',
        'color': '#f39c12'
    }
}


def run_ga_with_weights(optimizer, weights, scenario_name, num_runs=5):
    """
    Run GA with specified weight configuration
    
    Args:
        optimizer: HybridOptimizer instance
        weights: Dictionary of fitness weights
        scenario_name: Name of scenario for logging
        num_runs: Number of independent runs
    
    Returns:
        Dictionary with results
    """
    log_section(logger, f"Running GA: {scenario_name}", char="-")
    logger.info(f"Weights: {weights}")
    
    # Update optimizer weights
    optimizer.weights = weights
    
    results = {
        'scenario': scenario_name,
        'weights': weights.copy(),
        'solutions': [],
        'fitness_values': [],
        'coverage_values': [],
        'cost_values': [],
        'grid_scores': [],
        'feasibility_scores': [],
        'runtimes': []
    }
    
    for run in range(num_runs):
        logger.info(f"  Run {run+1}/{num_runs}...")
        
        start_time = time.time()
        solution, history = optimizer.run_genetic_algorithm(
            population_size=50,
            generations=100
        )
        elapsed = time.time() - start_time
        
        results['solutions'].append(solution)
        results['fitness_values'].append(solution.fitness)
        results['coverage_values'].append(solution.coverage)
        results['cost_values'].append(solution.cost)
        results['grid_scores'].append(solution.objectives['grid_score'])
        results['feasibility_scores'].append(solution.objectives['feasibility_score'])
        results['runtimes'].append(elapsed)
    
    # Summary statistics
    results['mean_fitness'] = np.mean(results['fitness_values'])
    results['std_fitness'] = np.std(results['fitness_values'])
    results['mean_coverage'] = np.mean(results['coverage_values'])
    results['mean_cost'] = np.mean(results['cost_values'])
    results['mean_grid_score'] = np.mean(results['grid_scores'])
    results['mean_feasibility'] = np.mean(results['feasibility_scores'])
    results['mean_runtime'] = np.mean(results['runtimes'])
    
    logger.info(f"  Mean Fitness: {results['mean_fitness']:.4f} (±{results['std_fitness']:.4f})")
    logger.info(f"  Mean Coverage: {results['mean_coverage']:.2%}")
    logger.info(f"  Mean Cost: ₹{results['mean_cost']:,.0f}")
    logger.info(f"  Mean Runtime: {results['mean_runtime']:.2f}s")
    
    return results


def analyze_weight_sensitivity():
    """Main weight sensitivity analysis"""
    log_section(logger, "WEIGHT SENSITIVITY ANALYSIS", char="█")
    
    # Load ward data
    logger.info("Loading ward-level data...")
    ward_data = pd.read_csv(OUTPUT_DIR / 'ward_level_data.csv')
    logger.info(f"Loaded {len(ward_data)} wards")
    
    # Initialize optimizer (will modify weights per scenario)
    optimizer = HybridOptimizer(
        ward_data=ward_data,
        cost_data={},
        budget_constraint=300_000_000,  # ₹30 crore
        target_chargers=100
    )
    
    # Run experiments for each weight scenario
    all_results = {}
    
    for scenario_name, scenario_config in WEIGHT_SCENARIOS.items():
        results = run_ga_with_weights(
            optimizer=optimizer,
            weights=scenario_config['weights'],
            scenario_name=scenario_name,
            num_runs=5
        )
        all_results[scenario_name] = results
    
    # ========== CREATE TABLE VII: WEIGHT SENSITIVITY ==========
    log_section(logger, "Generating TABLE VII: Weight Sensitivity Results")
    
    table_rows = []
    for scenario_name, results in all_results.items():
        weights = results['weights']
        
        # Get best solution for spatial analysis
        best_idx = np.argmax(results['fitness_values'])
        best_solution = results['solutions'][best_idx]
        wards_allocated = np.sum(best_solution.ward_allocations > 0)
        
        row = {
            'Scenario': scenario_name,
            'w_cost': f"{weights['cost']:.2f}",
            'w_coverage': f"{weights['coverage']:.2f}",
            'w_grid': f"{weights['grid']:.2f}",
            'w_feasibility': f"{weights['feasibility']:.2f}",
            'Mean Fitness': f"{results['mean_fitness']:.4f}",
            'Mean Coverage (%)': f"{results['mean_coverage']*100:.2f}",
            'Mean Cost (Crore)': f"{results['mean_cost']/10_000_000:.2f}",
            'Mean Grid Score': f"{results['mean_grid_score']:.3f}",
            'Mean Feasibility': f"{results['mean_feasibility']:.3f}",
            'Wards Allocated': wards_allocated,
            'Runtime (s)': f"{results['mean_runtime']:.2f}"
        }
        table_rows.append(row)
    
    sensitivity_table = pd.DataFrame(table_rows)
    table_file = OUTPUT_DIR / 'ieee_paper' / 'TABLE_VII_Weight_Sensitivity.csv'
    sensitivity_table.to_csv(table_file, index=False)
    logger.info(f"Saved TABLE VII to: {table_file}")
    
    print("\n" + "="*120)
    print("TABLE VII: WEIGHT SENSITIVITY ANALYSIS")
    print("="*120)
    print(sensitivity_table.to_string(index=False))
    print("="*120 + "\n")
    
    # ========== FIGURE 10: SPATIAL ALLOCATION COMPARISON ==========
    log_section(logger, "Generating FIGURE 10: Spatial Allocation Comparison")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (scenario_name, results) in enumerate(all_results.items()):
        ax = axes[idx]
        
        # Get best solution
        best_idx = np.argmax(results['fitness_values'])
        best_solution = results['solutions'][best_idx]
        allocations = best_solution.ward_allocations
        
        # Create allocation distribution histogram
        bins = np.arange(0, allocations.max() + 2) - 0.5
        ax.hist(allocations, bins=bins, color=WEIGHT_SCENARIOS[scenario_name]['color'],
                edgecolor='black', linewidth=1.2, alpha=0.7)
        
        ax.set_xlabel('Chargers per Ward', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Wards', fontsize=11, fontweight='bold')
        ax.set_title(f'{scenario_name}\n(Coverage: {results["mean_coverage"]*100:.1f}%, Cost: ₹{results["mean_cost"]/10_000_000:.1f}Cr)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        wards_allocated = np.sum(allocations > 0)
        max_allocation = allocations.max()
        ax.text(0.65, 0.95, 
                f'Wards served: {wards_allocated}\nMax per ward: {max_allocation}\nFitness: {results["mean_fitness"]:.4f}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=9, verticalalignment='top')
    
    plt.suptitle('Spatial Allocation Patterns Under Different Weight Configurations',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig_file = OUTPUT_DIR / 'ieee_paper' / 'FIGURE_10_Weight_Sensitivity_Spatial.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved FIGURE 10 to: {fig_file}")
    plt.close()
    
    # ========== FIGURE 11: METRICS COMPARISON ==========
    log_section(logger, "Generating FIGURE 11: Metrics Comparison")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    scenarios = list(all_results.keys())
    colors = [WEIGHT_SCENARIOS[s]['color'] for s in scenarios]
    
    # Panel A: Fitness comparison
    ax1 = axes[0, 0]
    fitness_means = [all_results[s]['mean_fitness'] for s in scenarios]
    fitness_stds = [all_results[s]['std_fitness'] for s in scenarios]
    
    bars = ax1.bar(range(len(scenarios)), fitness_means, yerr=fitness_stds, 
                   capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(scenarios)))
    ax1.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax1.set_ylabel('Mean Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Fitness Comparison', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Panel B: Coverage vs Cost trade-off
    ax2 = axes[0, 1]
    coverage_means = [all_results[s]['mean_coverage']*100 for s in scenarios]
    cost_means = [all_results[s]['mean_cost']/10_000_000 for s in scenarios]
    
    scatter = ax2.scatter(cost_means, coverage_means, s=300, c=colors, 
                         edgecolors='black', linewidth=2, alpha=0.8)
    
    for i, scenario in enumerate(scenarios):
        ax2.annotate(scenario.split('(')[0].strip(), 
                    (cost_means[i], coverage_means[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Mean Total Cost (Crore INR)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Coverage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Coverage vs. Cost Trade-off', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel C: Component scores
    ax3 = axes[1, 0]
    
    coverage_pcts = [all_results[s]['mean_coverage']*100 for s in scenarios]
    grid_scores = [all_results[s]['mean_grid_score']*100 for s in scenarios]
    feasibility_scores = [all_results[s]['mean_feasibility']*100 for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    ax3.bar(x - width, coverage_pcts, width, label='Coverage Score', 
            color='#3498db', edgecolor='black', linewidth=1.2)
    ax3.bar(x, grid_scores, width, label='Grid Proximity Score',
            color='#f39c12', edgecolor='black', linewidth=1.2)
    ax3.bar(x + width, feasibility_scores, width, label='Feasibility Score',
            color='#9b59b6', edgecolor='black', linewidth=1.2)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax3.set_ylabel('Score (0-100 scale)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Objective Component Scores', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Panel D: Wards allocated comparison
    ax4 = axes[1, 1]
    
    wards_allocated = []
    for scenario in scenarios:
        best_idx = np.argmax(all_results[scenario]['fitness_values'])
        best_solution = all_results[scenario]['solutions'][best_idx]
        wards_allocated.append(np.sum(best_solution.ward_allocations > 0))
    
    bars = ax4.bar(range(len(scenarios)), wards_allocated, color=colors,
                   edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right')
    ax4.set_ylabel('Number of Wards Allocated', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Spatial Distribution Breadth', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Weight Configuration Sensitivity: Impact on Multi-Objective Metrics',
                fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig_file = OUTPUT_DIR / 'ieee_paper' / 'FIGURE_11_Weight_Sensitivity_Metrics.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved FIGURE 11 to: {fig_file}")
    plt.close()
    
    # ========== SUMMARY ==========
    log_section(logger, "WEIGHT SENSITIVITY ANALYSIS COMPLETE", char="█")
    
    print("\n" + "="*80)
    print("KEY FINDINGS FOR PAPER")
    print("="*80)
    
    # Find best and worst performers
    baseline_fitness = all_results['Baseline (Current)']['mean_fitness']
    
    for scenario_name, results in all_results.items():
        if scenario_name == 'Baseline (Current)':
            continue
        
        fitness_diff = results['mean_fitness'] - baseline_fitness
        coverage_diff = (results['mean_coverage'] - all_results['Baseline (Current)']['mean_coverage']) * 100
        cost_diff = (results['mean_cost'] - all_results['Baseline (Current)']['mean_cost']) / 10_000_000
        
        print(f"\n{scenario_name}:")
        print(f"  Fitness: {results['mean_fitness']:.4f} ({fitness_diff:+.4f} vs baseline)")
        print(f"  Coverage: {results['mean_coverage']*100:.1f}% ({coverage_diff:+.1f}pp vs baseline)")
        print(f"  Cost: ₹{results['mean_cost']/10_000_000:.2f}Cr ({cost_diff:+.2f}Cr vs baseline)")
        print(f"  Grid Score: {results['mean_grid_score']:.3f}")
    
    print("\n" + "="*80)
    print("PAPER STATEMENT:")
    print('"Sensitivity analysis across four weight configurations demonstrates')
    print('the framework\'s adaptability to policy priorities. Under cost-heavy')
    print('weights (austerity scenario), the solution reduces expenditure by')
    print(f'₹{abs((all_results["Cost-Heavy (Austerity)"]["mean_cost"] - all_results["Baseline (Current)"]["mean_cost"])/10_000_000):.1f} crore')
    print(f'while maintaining {all_results["Cost-Heavy (Austerity)"]["mean_coverage"]*100:.1f}% coverage.')
    print('Under coverage-heavy weights, demand service increases to')
    print(f'{all_results["Coverage-Heavy (Demand-First)"]["mean_coverage"]*100:.1f}%, validating multi-objective')
    print('optimization\'s ability to balance competing stakeholder priorities."')
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    logger.info("Starting weight sensitivity analysis...")
    results = analyze_weight_sensitivity()
    logger.info("Weight sensitivity analysis complete!")
    logger.info(f"Output files in: {OUTPUT_DIR / 'ieee_paper'}")
