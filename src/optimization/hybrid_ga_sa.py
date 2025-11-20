"""
Hybrid Genetic Algorithm + Simulated Annealing for EV Charging Station Placement

Optimization objectives:
1. Minimize total cost (installation + operational + land rental)
2. Maximize demand coverage
3. Maximize grid proximity
4. Maximize land use feasibility

Approach:
- GA for global exploration and population-based search
- SA for local refinement of best solutions
- Multi-objective fitness with weighted scalarization
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List, Tuple, Dict
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import *
from src.utils.logger import setup_logger, log_section

logger = setup_logger(__name__)


class Solution:
    """Represents a candidate solution (charger placement)"""
    
    def __init__(self, ward_allocations: np.ndarray):
        """
        Args:
            ward_allocations: Array of charger counts per ward (length = num_wards)
        """
        self.ward_allocations = ward_allocations.copy()
        self.fitness = None
        self.cost = None
        self.coverage = None
        self.objectives = {}
    
    def copy(self):
        """Create a deep copy"""
        new_sol = Solution(self.ward_allocations)
        new_sol.fitness = self.fitness
        new_sol.cost = self.cost
        new_sol.coverage = self.coverage
        new_sol.objectives = self.objectives.copy()
        return new_sol
    
    def total_chargers(self) -> int:
        """Total number of chargers in solution"""
        return int(np.sum(self.ward_allocations))


class HybridOptimizer:
    """Hybrid GA+SA optimizer for charger placement"""
    
    def __init__(
        self,
        ward_data: pd.DataFrame,
        cost_data: Dict,
        budget_constraint: float = 50_000_000,  # 5 crore INR
        target_chargers: int = 100,
        charger_capacity_kwh: float = 50,  # kW fast charger
    ):
        """
        Initialize optimizer
        
        Args:
            ward_data: Ward-level aggregated data
            cost_data: Cost data dictionary
            budget_constraint: Maximum budget in INR
            target_chargers: Target number of chargers to deploy
            charger_capacity_kwh: Capacity per charger (kW)
        """
        self.ward_data = ward_data.copy()
        self.cost_data = cost_data
        self.budget_constraint = budget_constraint
        self.target_chargers = target_chargers
        self.charger_capacity_kwh = charger_capacity_kwh
        
        self.num_wards = len(ward_data)
        
        # Extract cost parameters (simplified - using average costs)
        self.installation_cost_per_charger = 1_500_000  # 15 lakh INR per fast charger
        self.annual_operational_cost = 200_000  # 2 lakh INR per charger per year
        self.land_rental_per_sqm_year = 5000  # Average rental
        self.charger_footprint_sqm = 20  # Space per charger
        
        # Fitness weights
        self.weights = {
            'cost': -0.30,       # Minimize cost (negative weight)
            'coverage': 0.40,    # Maximize coverage
            'grid': 0.15,        # Maximize grid proximity
            'feasibility': 0.15  # Maximize land feasibility
        }
        
        logger.info(f"HybridOptimizer initialized:")
        logger.info(f"  Wards: {self.num_wards}")
        logger.info(f"  Budget: ₹{budget_constraint:,.0f}")
        logger.info(f"  Target chargers: {target_chargers}")
        logger.info(f"  Fitness weights: {self.weights}")
    
    def calculate_fitness(self, solution: Solution) -> float:
        """
        Calculate multi-objective fitness score
        
        Components:
        1. Cost (installation + operational for 5 years + land rental)
        2. Demand coverage (% of citywide demand served)
        3. Grid proximity (wards with substations)
        4. Land use feasibility
        
        Args:
            solution: Candidate solution
        
        Returns:
            Fitness score (higher is better)
        """
        allocations = solution.ward_allocations
        
        # 1. Calculate total cost
        total_chargers = allocations.sum()
        installation_cost = total_chargers * self.installation_cost_per_charger
        operational_cost_5yr = total_chargers * self.annual_operational_cost * 5
        land_rental_5yr = total_chargers * self.charger_footprint_sqm * self.land_rental_per_sqm_year * 5
        total_cost = installation_cost + operational_cost_5yr + land_rental_5yr
        
        # Normalize cost (0-1, where 0 = budget, 1 = free)
        cost_normalized = 1 - min(total_cost / self.budget_constraint, 1.0)
        
        # 2. Calculate demand coverage
        # Chargers can serve capacity * utilization hours
        charger_daily_capacity_kwh = self.charger_capacity_kwh * 12  # 12 hours utilization
        
        ward_capacity = allocations * charger_daily_capacity_kwh
        ward_demand = self.ward_data['estimated_daily_demand_kwh'].values
        
        # Coverage: min(capacity, demand) / total_demand
        served_demand = np.minimum(ward_capacity, ward_demand)
        total_demand = ward_demand.sum()
        coverage_ratio = served_demand.sum() / total_demand if total_demand > 0 else 0
        
        # 3. Calculate grid proximity score
        has_grid = (self.ward_data['substation_count'] > 0).values.astype(float)
        grid_score = np.sum(allocations * has_grid) / total_chargers if total_chargers > 0 else 0
        
        # 4. Calculate feasibility score
        feasibility = self.ward_data['land_use_feasibility_score'].values
        feasibility_score = np.sum(allocations * feasibility) / total_chargers if total_chargers > 0 else 0
        
        # Composite fitness (weighted sum)
        fitness = (
            self.weights['cost'] * cost_normalized +
            self.weights['coverage'] * coverage_ratio +
            self.weights['grid'] * grid_score +
            self.weights['feasibility'] * feasibility_score
        )
        
        # Store components
        solution.fitness = fitness
        solution.cost = total_cost
        solution.coverage = coverage_ratio
        solution.objectives = {
            'total_cost': total_cost,
            'coverage_ratio': coverage_ratio,
            'grid_score': grid_score,
            'feasibility_score': feasibility_score,
            'cost_normalized': cost_normalized
        }
        
        return fitness
    
    def generate_initial_solution(self) -> Solution:
        """
        Generate random initial solution
        
        Strategy: Randomly distribute target chargers across wards,
        with bias towards high-demand, high-feasibility wards
        """
        # Probability proportional to demand * feasibility
        demand = self.ward_data['estimated_daily_demand_kwh'].values
        feasibility = self.ward_data['land_use_feasibility_score'].values
        
        # Avoid division by zero
        weights = (demand + 1) * (feasibility + 0.1)
        weights = weights / weights.sum()
        
        # Allocate chargers
        allocations = np.zeros(self.num_wards, dtype=int)
        for _ in range(self.target_chargers):
            ward_idx = np.random.choice(self.num_wards, p=weights)
            allocations[ward_idx] += 1
        
        solution = Solution(allocations)
        self.calculate_fitness(solution)
        return solution
    
    # ==================== GENETIC ALGORITHM ====================
    
    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        Uniform crossover: randomly swap ward allocations
        
        Args:
            parent1, parent2: Parent solutions
        
        Returns:
            Two offspring solutions
        """
        alloc1 = parent1.ward_allocations.copy()
        alloc2 = parent2.ward_allocations.copy()
        
        # Uniform crossover
        mask = np.random.rand(self.num_wards) < 0.5
        child1_alloc = np.where(mask, alloc1, alloc2)
        child2_alloc = np.where(mask, alloc2, alloc1)
        
        # Normalize to target charger count
        child1_alloc = self._normalize_allocation(child1_alloc)
        child2_alloc = self._normalize_allocation(child2_alloc)
        
        child1 = Solution(child1_alloc)
        child2 = Solution(child2_alloc)
        
        self.calculate_fitness(child1)
        self.calculate_fitness(child2)
        
        return child1, child2
    
    def mutate(self, solution: Solution, mutation_rate: float = 0.15) -> Solution:
        """
        Mutation: randomly move chargers between wards
        
        Args:
            solution: Solution to mutate
            mutation_rate: Probability of mutation
        
        Returns:
            Mutated solution
        """
        mutated = solution.copy()
        allocations = mutated.ward_allocations
        
        if np.random.rand() < mutation_rate:
            # Move chargers between random wards
            num_moves = max(1, int(self.target_chargers * 0.1))  # Move 10% of chargers
            
            for _ in range(num_moves):
                # Remove from random ward with chargers
                wards_with_chargers = np.where(allocations > 0)[0]
                if len(wards_with_chargers) > 0:
                    from_ward = np.random.choice(wards_with_chargers)
                    allocations[from_ward] -= 1
                    
                    # Add to random ward (weighted by demand/feasibility)
                    demand = self.ward_data['estimated_daily_demand_kwh'].values
                    feasibility = self.ward_data['land_use_feasibility_score'].values
                    weights = (demand + 1) * (feasibility + 0.1)
                    weights = weights / weights.sum()
                    
                    to_ward = np.random.choice(self.num_wards, p=weights)
                    allocations[to_ward] += 1
        
        mutated.ward_allocations = allocations
        self.calculate_fitness(mutated)
        return mutated
    
    def _normalize_allocation(self, allocations: np.ndarray) -> np.ndarray:
        """Normalize allocation to match target charger count"""
        current_total = allocations.sum()
        target = self.target_chargers
        
        if current_total == target:
            return allocations
        elif current_total < target:
            # Add chargers randomly
            diff = target - current_total
            for _ in range(diff):
                ward = np.random.randint(self.num_wards)
                allocations[ward] += 1
        else:
            # Remove chargers from wards with most chargers
            diff = current_total - target
            for _ in range(diff):
                wards_with_chargers = np.where(allocations > 0)[0]
                if len(wards_with_chargers) > 0:
                    ward = np.random.choice(wards_with_chargers)
                    allocations[ward] = max(0, allocations[ward] - 1)
        
        return allocations.astype(int)
    
    def run_genetic_algorithm(
        self,
        population_size: int = 50,
        generations: int = 100,
        elite_size: int = 5,
        mutation_rate: float = 0.15
    ) -> Tuple[Solution, List[float]]:
        """
        Run genetic algorithm
        
        Args:
            population_size: Number of solutions per generation
            generations: Number of generations
            elite_size: Number of top solutions to preserve
            mutation_rate: Probability of mutation
        
        Returns:
            Best solution and fitness history
        """
        log_section(logger, "RUNNING GENETIC ALGORITHM")
        logger.info(f"Population: {population_size}, Generations: {generations}")
        logger.info(f"Elite: {elite_size}, Mutation rate: {mutation_rate}")
        
        # Initialize population
        population = [self.generate_initial_solution() for _ in range(population_size)]
        
        fitness_history = []
        best_solution = max(population, key=lambda s: s.fitness)
        
        for gen in range(generations):
            # Sort by fitness
            population.sort(key=lambda s: s.fitness, reverse=True)
            
            # Track best fitness
            current_best = population[0]
            fitness_history.append(current_best.fitness)
            
            if current_best.fitness > best_solution.fitness:
                best_solution = current_best.copy()
            
            # Log progress
            if gen % 20 == 0 or gen == generations - 1:
                logger.info(
                    f"Gen {gen:3d}: Best fitness={current_best.fitness:.4f}, "
                    f"Coverage={current_best.coverage:.2%}, Cost=₹{current_best.cost:,.0f}"
                )
            
            # Elitism: preserve top solutions
            next_population = population[:elite_size]
            
            # Generate offspring via crossover and mutation
            while len(next_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(population, k=3)
                parent2 = self._tournament_select(population, k=3)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1, mutation_rate)
                child2 = self.mutate(child2, mutation_rate)
                
                next_population.extend([child1, child2])
            
            population = next_population[:population_size]
        
        log_section(logger, "GENETIC ALGORITHM COMPLETE")
        logger.info(f"Best fitness: {best_solution.fitness:.4f}")
        logger.info(f"Coverage: {best_solution.coverage:.2%}")
        logger.info(f"Total cost: ₹{best_solution.cost:,.0f}")
        
        return best_solution, fitness_history
    
    def _tournament_select(self, population: List[Solution], k: int = 3) -> Solution:
        """Tournament selection"""
        tournament = random.sample(population, k)
        return max(tournament, key=lambda s: s.fitness)
    
    # ==================== SIMULATED ANNEALING ====================
    
    def run_simulated_annealing(
        self,
        initial_solution: Solution,
        initial_temp: float = 1000.0,
        cooling_rate: float = 0.95,
        iterations_per_temp: int = 50,
        min_temp: float = 1.0
    ) -> Tuple[Solution, List[float]]:
        """
        Run simulated annealing for local refinement
        
        Args:
            initial_solution: Starting solution (typically from GA)
            initial_temp: Initial temperature
            cooling_rate: Temperature reduction factor
            iterations_per_temp: Iterations at each temperature
            min_temp: Minimum temperature (stopping criterion)
        
        Returns:
            Refined solution and fitness history
        """
        log_section(logger, "RUNNING SIMULATED ANNEALING REFINEMENT")
        logger.info(f"Initial temp: {initial_temp}, Cooling: {cooling_rate}")
        logger.info(f"Iterations per temp: {iterations_per_temp}")
        
        current = initial_solution.copy()
        best = current.copy()
        
        temp = initial_temp
        fitness_history = [current.fitness]
        iteration = 0
        
        while temp > min_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor solution (small perturbation)
                neighbor = self._generate_neighbor(current)
                
                # Calculate acceptance probability
                delta = neighbor.fitness - current.fitness
                
                if delta > 0:
                    # Better solution - always accept
                    current = neighbor.copy()
                    if current.fitness > best.fitness:
                        best = current.copy()
                else:
                    # Worse solution - accept with probability
                    acceptance_prob = np.exp(delta / temp)
                    if np.random.rand() < acceptance_prob:
                        current = neighbor.copy()
                
                fitness_history.append(current.fitness)
                iteration += 1
            
            # Cool down
            temp *= cooling_rate
            
            if iteration % 500 == 0:
                logger.info(
                    f"Iter {iteration:4d}, Temp={temp:6.1f}: "
                    f"Best fitness={best.fitness:.4f}, Coverage={best.coverage:.2%}"
                )
        
        log_section(logger, "SIMULATED ANNEALING COMPLETE")
        logger.info(f"Final fitness: {best.fitness:.4f}")
        logger.info(f"Improvement: {best.fitness - initial_solution.fitness:.4f}")
        logger.info(f"Coverage: {best.coverage:.2%}")
        
        return best, fitness_history
    
    def _generate_neighbor(self, solution: Solution) -> Solution:
        """
        Generate neighbor solution for SA
        
        Strategy: Small random perturbation (move 1-3 chargers)
        """
        neighbor = solution.copy()
        allocations = neighbor.ward_allocations
        
        # Move 1-3 chargers
        num_moves = np.random.randint(1, 4)
        
        for _ in range(num_moves):
            # Remove from random ward
            wards_with_chargers = np.where(allocations > 0)[0]
            if len(wards_with_chargers) > 0:
                from_ward = np.random.choice(wards_with_chargers)
                allocations[from_ward] -= 1
                
                # Add to adjacent or random ward
                to_ward = np.random.randint(self.num_wards)
                allocations[to_ward] += 1
        
        neighbor.ward_allocations = allocations
        self.calculate_fitness(neighbor)
        return neighbor
    
    # ==================== HYBRID WORKFLOW ====================
    
    def optimize(
        self,
        ga_generations: int = 100,
        ga_population: int = 50,
        sa_refinement: bool = True,
        sa_iterations: int = 2000
    ) -> Dict:
        """
        Run hybrid GA+SA optimization
        
        Workflow:
        1. Run GA to explore solution space
        2. Apply SA to best GA solution for local refinement
        
        Args:
            ga_generations: GA generations
            ga_population: GA population size
            sa_refinement: Whether to apply SA refinement
            sa_iterations: Total SA iterations
        
        Returns:
            Dictionary with results
        """
        log_section(logger, "HYBRID GA+SA OPTIMIZATION", char="*")
        
        # Phase 1: Genetic Algorithm
        ga_solution, ga_history = self.run_genetic_algorithm(
            population_size=ga_population,
            generations=ga_generations
        )
        
        # Phase 2: Simulated Annealing (optional)
        if sa_refinement:
            final_solution, sa_history = self.run_simulated_annealing(
                initial_solution=ga_solution,
                iterations_per_temp=sa_iterations // 20
            )
            combined_history = ga_history + sa_history
        else:
            final_solution = ga_solution
            combined_history = ga_history
        
        # Compile results
        results = {
            'solution': final_solution,
            'fitness_history': combined_history,
            'ga_solution': ga_solution,
            'allocations': final_solution.ward_allocations,
            'ward_data': self.ward_data,
            'metrics': {
                'total_chargers': final_solution.total_chargers(),
                'total_cost': final_solution.cost,
                'coverage_ratio': final_solution.coverage,
                'fitness': final_solution.fitness,
                **final_solution.objectives
            }
        }
        
        log_section(logger, "OPTIMIZATION COMPLETE", char="*")
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print optimization results"""
        metrics = results['metrics']
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Total chargers deployed: {metrics['total_chargers']}")
        print(f"Total cost: ₹{metrics['total_cost']:,.0f} ({metrics['total_cost']/10_000_000:.2f} crore)")
        print(f"Budget utilization: {metrics['total_cost']/self.budget_constraint*100:.1f}%")
        print(f"Demand coverage: {metrics['coverage_ratio']:.2%}")
        print(f"Grid proximity score: {metrics['grid_score']:.3f}")
        print(f"Feasibility score: {metrics['feasibility_score']:.3f}")
        print(f"Overall fitness: {metrics['fitness']:.4f}")
        
        # Wards with chargers
        allocations = results['allocations']
        wards_with_chargers = np.where(allocations > 0)[0]
        print(f"\nChargers deployed in {len(wards_with_chargers)} wards")
        print("="*80 + "\n")


def plot_optimization_results(results: Dict, output_dir: Path):
    """Create visualization of optimization results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Fitness convergence
    ax1 = axes[0, 0]
    history = results['fitness_history']
    ax1.plot(history, linewidth=2, color='steelblue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Optimization Convergence (GA + SA)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=results['ga_solution'].fitness, color='red', linestyle='--', label='GA Best')
    ax1.legend()
    
    # 2. Charger allocation distribution
    ax2 = axes[0, 1]
    allocations = results['allocations']
    allocation_counts = pd.Series(allocations).value_counts().sort_index()
    ax2.bar(allocation_counts.index, allocation_counts.values, color='orange', edgecolor='black')
    ax2.set_xlabel('Chargers per Ward')
    ax2.set_ylabel('Number of Wards')
    ax2.set_title('Distribution of Allocated Chargers')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 15 wards by allocation
    ax3 = axes[1, 0]
    ward_data = results['ward_data'].copy()
    ward_data['allocated_chargers'] = allocations
    top_wards = ward_data.nlargest(15, 'allocated_chargers')[['KGISWardName', 'allocated_chargers']]
    ax3.barh(range(len(top_wards)), top_wards['allocated_chargers'].values, color='green')
    ax3.set_yticks(range(len(top_wards)))
    ax3.set_yticklabels(top_wards['KGISWardName'].values, fontsize=8)
    ax3.set_xlabel('Number of Chargers')
    ax3.set_title('Top 15 Wards by Charger Allocation')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Metrics breakdown
    ax4 = axes[1, 1]
    metrics = results['metrics']
    metric_names = ['Coverage\nRatio', 'Grid\nScore', 'Feasibility\nScore']
    metric_values = [
        metrics['coverage_ratio'],
        metrics['grid_score'],
        metrics['feasibility_score']
    ]
    colors = ['#2ecc71', '#3498db', '#f39c12']
    bars = ax4.bar(metric_names, metric_values, color=colors, edgecolor='black')
    ax4.set_ylabel('Score (0-1)')
    ax4.set_title('Objective Components')
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'optimization_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved optimization visualization to {output_file}")
    plt.close()


if __name__ == "__main__":
    logger.info("Loading ward-level data...")
    
    # Load data
    ward_data = pd.read_csv(OUTPUT_DIR / 'ward_level_data.csv')
    
    # Simplified cost data (would normally load from files)
    cost_data = {}
    
    # Initialize optimizer
    optimizer = HybridOptimizer(
        ward_data=ward_data,
        cost_data=cost_data,
        budget_constraint=50_000_000,  # 5 crore
        target_chargers=100
    )
    
    # Run optimization
    results = optimizer.optimize(
        ga_generations=100,
        ga_population=50,
        sa_refinement=True,
        sa_iterations=2000
    )
    
    # Save results
    output_file = OUTPUT_DIR / 'optimized_allocation.csv'
    allocation_df = ward_data[['KGISWardID', 'KGISWardName']].copy()
    allocation_df['allocated_chargers'] = results['allocations']
    allocation_df['estimated_demand_kwh'] = ward_data['estimated_daily_demand_kwh']
    allocation_df['existing_chargers'] = ward_data['existing_chargers_count']
    allocation_df = allocation_df[allocation_df['allocated_chargers'] > 0].sort_values('allocated_chargers', ascending=False)
    allocation_df.to_csv(output_file, index=False)
    logger.info(f"Saved allocation to {output_file}")
    
    # Visualize
    plot_optimization_results(results, OUTPUT_DIR)
    
    logger.info("Optimization complete!")
