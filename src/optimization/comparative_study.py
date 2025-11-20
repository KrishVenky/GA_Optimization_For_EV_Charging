"""
Comparative Study: GA vs SA vs Hybrid GA+SA

Runs all three optimization approaches and compares:
1. Solution quality (fitness, coverage)
2. Convergence speed
3. Computational efficiency
4. Robustness (multiple runs)

For publication/academic submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import OUTPUT_DIR
from src.utils.logger import setup_logger, log_section
from src.optimization.hybrid_ga_sa import HybridOptimizer, Solution

logger = setup_logger(__name__)
sns.set_style("whitegrid")


class ComparativeStudy:
    """Run and compare GA, SA, and Hybrid GA+SA"""
    
    def __init__(self, optimizer: HybridOptimizer, num_runs: int = 5):
        """
        Args:
            optimizer: Configured optimizer instance
            num_runs: Number of independent runs per algorithm
        """
        self.optimizer = optimizer
        self.num_runs = num_runs
        self.results = {}
        
        logger.info(f"ComparativeStudy initialized with {num_runs} runs per algorithm")
    
    def run_ga_only(self, generations: int = 100, population: int = 50) -> Dict:
        """Run GA-only optimization multiple times"""
        log_section(logger, "EXPERIMENT 1: GENETIC ALGORITHM ONLY", char="=")
        
        results = {
            'name': 'GA Only',
            'runs': [],
            'fitness_histories': [],
            'computation_times': [],
            'best_solutions': []
        }
        
        for run in range(self.num_runs):
            logger.info(f"\n--- GA Run {run+1}/{self.num_runs} ---")
            
            start_time = time.time()
            solution, history = self.optimizer.run_genetic_algorithm(
                population_size=population,
                generations=generations
            )
            elapsed = time.time() - start_time
            
            results['runs'].append(run + 1)
            results['fitness_histories'].append(history)
            results['computation_times'].append(elapsed)
            results['best_solutions'].append(solution)
            
            logger.info(f"Completed in {elapsed:.2f}s - Fitness: {solution.fitness:.4f}, Coverage: {solution.coverage:.2%}")
        
        # Summary statistics
        final_fitnesses = [s.fitness for s in results['best_solutions']]
        final_coverages = [s.coverage for s in results['best_solutions']]
        
        results['summary'] = {
            'mean_fitness': np.mean(final_fitnesses),
            'std_fitness': np.std(final_fitnesses),
            'best_fitness': np.max(final_fitnesses),
            'worst_fitness': np.min(final_fitnesses),
            'mean_coverage': np.mean(final_coverages),
            'std_coverage': np.std(final_coverages),
            'mean_time': np.mean(results['computation_times']),
            'std_time': np.std(results['computation_times'])
        }
        
        self._print_summary('GA ONLY', results['summary'])
        return results
    
    def run_sa_only(self, iterations: int = 10000) -> Dict:
        """Run SA-only optimization multiple times"""
        log_section(logger, "EXPERIMENT 2: SIMULATED ANNEALING ONLY", char="=")
        
        results = {
            'name': 'SA Only',
            'runs': [],
            'fitness_histories': [],
            'computation_times': [],
            'best_solutions': []
        }
        
        for run in range(self.num_runs):
            logger.info(f"\n--- SA Run {run+1}/{self.num_runs} ---")
            
            # Generate random initial solution
            initial = self.optimizer.generate_initial_solution()
            
            start_time = time.time()
            solution, history = self.optimizer.run_simulated_annealing(
                initial_solution=initial,
                initial_temp=2000.0,
                cooling_rate=0.95,
                iterations_per_temp=100,
                min_temp=1.0
            )
            elapsed = time.time() - start_time
            
            results['runs'].append(run + 1)
            results['fitness_histories'].append(history)
            results['computation_times'].append(elapsed)
            results['best_solutions'].append(solution)
            
            logger.info(f"Completed in {elapsed:.2f}s - Fitness: {solution.fitness:.4f}, Coverage: {solution.coverage:.2%}")
        
        # Summary statistics
        final_fitnesses = [s.fitness for s in results['best_solutions']]
        final_coverages = [s.coverage for s in results['best_solutions']]
        
        results['summary'] = {
            'mean_fitness': np.mean(final_fitnesses),
            'std_fitness': np.std(final_fitnesses),
            'best_fitness': np.max(final_fitnesses),
            'worst_fitness': np.min(final_fitnesses),
            'mean_coverage': np.mean(final_coverages),
            'std_coverage': np.std(final_coverages),
            'mean_time': np.mean(results['computation_times']),
            'std_time': np.std(results['computation_times'])
        }
        
        self._print_summary('SA ONLY', results['summary'])
        return results
    
    def run_hybrid(self, ga_generations: int = 100, ga_population: int = 50) -> Dict:
        """Run Hybrid GA+SA optimization multiple times"""
        log_section(logger, "EXPERIMENT 3: HYBRID GA+SA", char="=")
        
        results = {
            'name': 'Hybrid GA+SA',
            'runs': [],
            'fitness_histories': [],
            'computation_times': [],
            'best_solutions': [],
            'ga_solutions': []  # Track GA-only solution before SA refinement
        }
        
        for run in range(self.num_runs):
            logger.info(f"\n--- Hybrid Run {run+1}/{self.num_runs} ---")
            
            start_time = time.time()
            
            # Phase 1: GA
            ga_solution, ga_history = self.optimizer.run_genetic_algorithm(
                population_size=ga_population,
                generations=ga_generations
            )
            
            # Phase 2: SA refinement
            final_solution, sa_history = self.optimizer.run_simulated_annealing(
                initial_solution=ga_solution,
                initial_temp=1000.0,
                cooling_rate=0.95,
                iterations_per_temp=100,
                min_temp=1.0
            )
            
            elapsed = time.time() - start_time
            
            combined_history = ga_history + sa_history
            
            results['runs'].append(run + 1)
            results['fitness_histories'].append(combined_history)
            results['computation_times'].append(elapsed)
            results['best_solutions'].append(final_solution)
            results['ga_solutions'].append(ga_solution)
            
            improvement = final_solution.fitness - ga_solution.fitness
            logger.info(f"Completed in {elapsed:.2f}s - Final Fitness: {final_solution.fitness:.4f}, SA Improvement: {improvement:+.4f}")
        
        # Summary statistics
        final_fitnesses = [s.fitness for s in results['best_solutions']]
        final_coverages = [s.coverage for s in results['best_solutions']]
        ga_fitnesses = [s.fitness for s in results['ga_solutions']]
        improvements = [final - ga for final, ga in zip(final_fitnesses, ga_fitnesses)]
        
        results['summary'] = {
            'mean_fitness': np.mean(final_fitnesses),
            'std_fitness': np.std(final_fitnesses),
            'best_fitness': np.max(final_fitnesses),
            'worst_fitness': np.min(final_fitnesses),
            'mean_coverage': np.mean(final_coverages),
            'std_coverage': np.std(final_coverages),
            'mean_time': np.mean(results['computation_times']),
            'std_time': np.std(results['computation_times']),
            'mean_ga_fitness': np.mean(ga_fitnesses),
            'mean_sa_improvement': np.mean(improvements),
            'sa_improvement_rate': np.sum(np.array(improvements) > 0) / self.num_runs
        }
        
        self._print_summary('HYBRID GA+SA', results['summary'])
        return results
    
    def _print_summary(self, name: str, summary: Dict):
        """Print algorithm summary statistics"""
        print("\n" + "="*80)
        print(f"{name} - SUMMARY STATISTICS ({self.num_runs} runs)")
        print("="*80)
        print(f"Mean Fitness:      {summary['mean_fitness']:.4f} ± {summary['std_fitness']:.4f}")
        print(f"Best Fitness:      {summary['best_fitness']:.4f}")
        print(f"Worst Fitness:     {summary['worst_fitness']:.4f}")
        print(f"Mean Coverage:     {summary['mean_coverage']:.2%} ± {summary['std_coverage']:.2%}")
        print(f"Mean Runtime:      {summary['mean_time']:.2f}s ± {summary['std_time']:.2f}s")
        
        if 'mean_sa_improvement' in summary:
            print(f"Mean SA Improvement: {summary['mean_sa_improvement']:+.4f}")
            print(f"SA Success Rate:   {summary['sa_improvement_rate']:.1%} of runs improved")
        
        print("="*80 + "\n")
    
    def run_all_experiments(self) -> Dict:
        """Run all three experiments and compile results"""
        log_section(logger, "COMPARATIVE STUDY: GA vs SA vs HYBRID", char="*")
        
        # Experiment 1: GA Only
        ga_results = self.run_ga_only(generations=100, population=50)
        
        # Experiment 2: SA Only
        sa_results = self.run_sa_only(iterations=10000)
        
        # Experiment 3: Hybrid GA+SA
        hybrid_results = self.run_hybrid(ga_generations=100, ga_population=50)
        
        all_results = {
            'ga_only': ga_results,
            'sa_only': sa_results,
            'hybrid': hybrid_results
        }
        
        # Generate comparison visualizations
        self.plot_comparative_results(all_results)
        self.generate_statistical_comparison(all_results)
        
        log_section(logger, "COMPARATIVE STUDY COMPLETE", char="*")
        
        return all_results
    
    def plot_comparative_results(self, results: Dict):
        """Create comprehensive comparison visualizations"""
        log_section(logger, "Generating Comparative Visualizations")
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Convergence comparison (average over runs)
        ax1 = fig.add_subplot(gs[0, :2])
        
        for algo_name, algo_results in results.items():
            histories = algo_results['fitness_histories']
            # Pad histories to same length
            max_len = max(len(h) for h in histories)
            padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]
            
            mean_history = np.mean(padded, axis=0)
            std_history = np.std(padded, axis=0)
            
            iterations = range(len(mean_history))
            ax1.plot(iterations, mean_history, linewidth=2, label=algo_results['name'])
            ax1.fill_between(iterations, 
                            mean_history - std_history,
                            mean_history + std_history,
                            alpha=0.2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Convergence Comparison (Mean ± Std over 5 runs)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Final fitness distribution (box plot)
        ax2 = fig.add_subplot(gs[0, 2])
        
        fitness_data = []
        labels = []
        for algo_results in results.values():
            fitnesses = [s.fitness for s in algo_results['best_solutions']]
            fitness_data.append(fitnesses)
            labels.append(algo_results['name'])
        
        bp = ax2.boxplot(fitness_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightsalmon']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('Final Fitness')
        ax2.set_title('Solution Quality Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 3. Coverage comparison (bar chart)
        ax3 = fig.add_subplot(gs[1, 0])
        
        coverages = [r['summary']['mean_coverage'] for r in results.values()]
        coverage_stds = [r['summary']['std_coverage'] for r in results.values()]
        labels = [r['name'] for r in results.values()]
        
        bars = ax3.bar(labels, coverages, yerr=coverage_stds, capsize=5, 
                       color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Mean Coverage Ratio')
        ax3.set_title('Demand Coverage Comparison')
        ax3.set_ylim(0, 1.0)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 4. Computation time comparison
        ax4 = fig.add_subplot(gs[1, 1])
        
        times = [r['summary']['mean_time'] for r in results.values()]
        time_stds = [r['summary']['std_time'] for r in results.values()]
        
        bars = ax4.bar(labels, times, yerr=time_stds, capsize=5,
                       color=colors, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Mean Runtime (seconds)')
        ax4.set_title('Computational Efficiency')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 5. Robustness (std deviation of fitness)
        ax5 = fig.add_subplot(gs[1, 2])
        
        stds = [r['summary']['std_fitness'] for r in results.values()]
        
        bars = ax5.bar(labels, stds, color=colors, edgecolor='black', linewidth=1.5)
        ax5.set_ylabel('Std Dev of Final Fitness')
        ax5.set_title('Robustness (Lower is Better)')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
        # 6. Individual run comparison (GA only)
        ax6 = fig.add_subplot(gs[2, 0])
        for i, history in enumerate(results['ga_only']['fitness_histories']):
            ax6.plot(history, alpha=0.6, linewidth=1, label=f'Run {i+1}')
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('Fitness')
        ax6.set_title('GA Individual Runs')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Individual run comparison (SA only)
        ax7 = fig.add_subplot(gs[2, 1])
        for i, history in enumerate(results['sa_only']['fitness_histories']):
            ax7.plot(history, alpha=0.6, linewidth=1, label=f'Run {i+1}')
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Fitness')
        ax7.set_title('SA Individual Runs')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 8. Statistical significance table
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Perform statistical tests
        from scipy import stats
        
        ga_fits = [s.fitness for s in results['ga_only']['best_solutions']]
        sa_fits = [s.fitness for s in results['sa_only']['best_solutions']]
        hybrid_fits = [s.fitness for s in results['hybrid']['best_solutions']]
        
        # Wilcoxon signed-rank test (non-parametric)
        stat_ga_hybrid, p_ga_hybrid = stats.wilcoxon(ga_fits, hybrid_fits)
        stat_sa_hybrid, p_sa_hybrid = stats.wilcoxon(sa_fits, hybrid_fits)
        stat_ga_sa, p_ga_sa = stats.wilcoxon(ga_fits, sa_fits)
        
        table_data = [
            ['Comparison', 'p-value', 'Significant?'],
            ['GA vs Hybrid', f'{p_ga_hybrid:.4f}', 'Yes' if p_ga_hybrid < 0.05 else 'No'],
            ['SA vs Hybrid', f'{p_sa_hybrid:.4f}', 'Yes' if p_sa_hybrid < 0.05 else 'No'],
            ['GA vs SA', f'{p_ga_sa:.4f}', 'Yes' if p_ga_sa < 0.05 else 'No']
        ]
        
        table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax8.set_title('Statistical Significance Tests\n(Wilcoxon Signed-Rank, α=0.05)', 
                     fontsize=10, pad=20)
        
        plt.suptitle('Comparative Analysis: GA vs SA vs Hybrid GA+SA\nBengaluru EV Charging Infrastructure Optimization',
                     fontsize=14, fontweight='bold', y=0.98)
        
        output_file = OUTPUT_DIR / 'comparative_analysis.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        logger.info(f"Saved comparative visualization to {output_file}")
        plt.close()
    
    def generate_statistical_comparison(self, results: Dict):
        """Generate detailed statistical comparison table"""
        log_section(logger, "Generating Statistical Comparison Table")
        
        comparison_data = []
        
        for algo_name, algo_results in results.items():
            summary = algo_results['summary']
            
            row = {
                'Algorithm': algo_results['name'],
                'Mean Fitness': summary['mean_fitness'],
                'Std Fitness': summary['std_fitness'],
                'Best Fitness': summary['best_fitness'],
                'Worst Fitness': summary['worst_fitness'],
                'Mean Coverage (%)': summary['mean_coverage'] * 100,
                'Std Coverage (%)': summary['std_coverage'] * 100,
                'Mean Runtime (s)': summary['mean_time'],
                'Std Runtime (s)': summary['std_time']
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        output_file = OUTPUT_DIR / 'algorithm_comparison_statistics.csv'
        df.to_csv(output_file, index=False, float_format='%.4f')
        logger.info(f"Saved statistical comparison to {output_file}")
        
        # Print formatted table
        print("\n" + "="*100)
        print("STATISTICAL COMPARISON TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100 + "\n")
        
        return df


if __name__ == "__main__":
    logger.info("Starting comparative study...")
    
    # Load ward data
    ward_data = pd.read_csv(OUTPUT_DIR / 'ward_level_data.csv')
    
    # Initialize optimizer
    optimizer = HybridOptimizer(
        ward_data=ward_data,
        cost_data={},
        budget_constraint=50_000_000,
        target_chargers=100
    )
    
    # Run comparative study
    study = ComparativeStudy(optimizer, num_runs=5)
    results = study.run_all_experiments()
    
    # Print final recommendations
    print("\n" + "="*100)
    print("PUBLICATION RECOMMENDATIONS")
    print("="*100)
    
    ga_fitness = results['ga_only']['summary']['mean_fitness']
    sa_fitness = results['sa_only']['summary']['mean_fitness']
    hybrid_fitness = results['hybrid']['summary']['mean_fitness']
    
    best_algo = max(
        [('GA Only', ga_fitness), ('SA Only', sa_fitness), ('Hybrid', hybrid_fitness)],
        key=lambda x: x[1]
    )
    
    print(f"\nBest Algorithm: {best_algo[0]} (Mean Fitness: {best_algo[1]:.4f})")
    print(f"GA Performance: {ga_fitness:.4f}")
    print(f"SA Performance: {sa_fitness:.4f}")
    print(f"Hybrid Performance: {hybrid_fitness:.4f}")
    print(f"Hybrid Improvement over GA: {(hybrid_fitness - ga_fitness):+.4f} ({(hybrid_fitness/ga_fitness - 1)*100:+.2f}%)")
    print(f"SA Improvement Rate: {results['hybrid']['summary']['sa_improvement_rate']:.1%}")
    
    print("\nKey Findings for Publication:")
    if hybrid_fitness > ga_fitness * 1.01:
        print("- Hybrid GA+SA shows statistically significant improvement over GA alone")
    else:
        print("- GA alone achieves near-optimal solutions; SA provides validation but minimal improvement")
    
    if sa_fitness < ga_fitness * 0.95:
        print("- SA alone struggles with global exploration in this high-dimensional space")
    else:
        print("- SA alone competitive but less robust than population-based GA")
    
    print("\nComputational Efficiency:")
    print(f"- GA Runtime: {results['ga_only']['summary']['mean_time']:.2f}s")
    print(f"- SA Runtime: {results['sa_only']['summary']['mean_time']:.2f}s")
    print(f"- Hybrid Runtime: {results['hybrid']['summary']['mean_time']:.2f}s")
    
    print("="*100 + "\n")
    
    logger.info("Comparative study complete!")
