import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pydantic import BaseModel


class MethodStats(BaseModel):
    p50: float
    p95: float
    mean: float
    std: float
    data: List[float]

class ComparisonResults(BaseModel):
    better_p50: str
    better_p95: str
    better_mean: str
    p50_improvement: float
    p95_improvement: float
    p50_improvement_percent: float
    p95_improvement_percent: float

class PerformanceAnalysisResult(BaseModel):
    metric_name: str
    method1_name: str
    method2_name: str
    stable_epochs_analyzed: int
    method1: MethodStats
    method2: MethodStats
    comparison: ComparisonResults


class PercentileScoreCalculator:
    """
    A class to analyze and compare agent performance between different methods (e.g., Adaptiq vs GPT)
    using p50 and p95 percentiles with validation and error handling.
    """
    
    def __init__(self, stable_epoch_start: int = 50):
        """
        Initialize the analyzer.
        
        Args:
            stable_epoch_start: The epoch from which to consider the agent as "optimized" (0-indexed)
        """
        self.stable_epoch_start = stable_epoch_start
        self.results = {}
        
    def analyze_performance(self, 
                          adaptiq_data: List[float], 
                          gpt_data: List[float],
                          adaptiq_method: str = "Adaptiq",
                          gpt_method: str = "GPT",
                          metric_name: str = "Latency") -> PerformanceAnalysisResult:
        """
        Analyze performance data for two methods and calculate p50/p95 percentiles.
        
        Args:
            adaptiq_data: List of performance values for Adaptiq method
            gpt_data: List of performance values for GPT method
            adaptiq_method: Name of Adaptiq method for labeling
            gpt_method: Name of GPT method for labeling
            metric_name: Name of the metric being measured (e.g., "Latency", "Tokens")
            
        Returns:
            PerformanceAnalysisResult containing analysis results
            
        Raises:
            ValueError: If data is insufficient or invalid
        """
        
        # Validation: Check if data lists are not empty
        if not adaptiq_data or not gpt_data:
            raise ValueError("Both adaptiq_data and gpt_data must be non-empty lists")
        
        # Validation: Check if we have enough data after stable_epoch_start
        if len(adaptiq_data) <= self.stable_epoch_start or len(gpt_data) <= self.stable_epoch_start:
            # Auto-adjust stable_epoch_start if data is too small
            original_start = self.stable_epoch_start
            self.stable_epoch_start = max(0, min(len(adaptiq_data), len(gpt_data)) // 3)
            print(f"Warning: Insufficient data for stable_epoch_start={original_start}. "
                  f"Auto-adjusted to {self.stable_epoch_start}")
        
        # Step 1: Isolate stable performance data
        stable_method1 = adaptiq_data[self.stable_epoch_start:]
        stable_method2 = gpt_data[self.stable_epoch_start:]
        
        # Additional validation: Ensure we have at least 1 data point
        if len(stable_method1) == 0 or len(stable_method2) == 0:
            raise ValueError(f"No data available after epoch {self.stable_epoch_start}. "
                           f"Reduce stable_epoch_start or provide more data.")
        
        # Step 2: Calculate percentiles (handle edge case of single data point)
        if len(stable_method1) == 1:
            method1_p50 = method1_p95 = stable_method1[0]
            print(f"Warning: Only 1 data point for {adaptiq_method} after epoch {self.stable_epoch_start}. "
                  f"p50 and p95 will be the same.")
        else:
            method1_p50 = np.percentile(stable_method1, 50)
            method1_p95 = np.percentile(stable_method1, 95)
            
        if len(stable_method2) == 1:
            method2_p50 = method2_p95 = stable_method2[0]
            print(f"Warning: Only 1 data point for {gpt_method} after epoch {self.stable_epoch_start}. "
                  f"p50 and p95 will be the same.")
        else:
            method2_p50 = np.percentile(stable_method2, 50)
            method2_p95 = np.percentile(stable_method2, 95)
        
        # Step 3: Calculate additional statistics
        method1_mean = np.mean(stable_method1)
        method1_std = np.std(stable_method1)
        method2_mean = np.mean(stable_method2)
        method2_std = np.std(stable_method2)
        
        # Step 4: Determine which method is better (lower is better for latency/tokens)
        better_p50 = gpt_method if method2_p50 < method1_p50 else adaptiq_method
        better_p95 = gpt_method if method2_p95 < method1_p95 else adaptiq_method
        better_mean = gpt_method if method2_mean < method1_mean else adaptiq_method
        
        # Calculate improvements with zero-division protection
        p50_improvement = abs(method1_p50 - method2_p50)
        p95_improvement = abs(method1_p95 - method2_p95)
        p50_improvement_percent = p50_improvement / max(method1_p50, method2_p50) * 100 if max(method1_p50, method2_p50) > 0 else 0
        p95_improvement_percent = p95_improvement / max(method1_p95, method2_p95) * 100 if max(method1_p95, method2_p95) > 0 else 0
        
        # Store results
        results = {
            'metric_name': metric_name,
            'method1_name': adaptiq_method,
            'method2_name': gpt_method,
            'stable_epochs_analyzed': len(stable_method1),
            'method1': {
                'p50': method1_p50,
                'p95': method1_p95,
                'mean': method1_mean,
                'std': method1_std,
                'data': stable_method1
            },
            'method2': {
                'p50': method2_p50,
                'p95': method2_p95,
                'mean': method2_mean,
                'std': method2_std,
                'data': stable_method2
            },
            'comparison': {
                'better_p50': better_p50,
                'better_p95': better_p95,
                'better_mean': better_mean,
                'p50_improvement': p50_improvement,
                'p95_improvement': p95_improvement,
                'p50_improvement_percent': p50_improvement_percent,
                'p95_improvement_percent': p95_improvement_percent
            }
        }
        
        result = PerformanceAnalysisResult(**results)
        self.results[metric_name] = result
        
        return result
    
    def plot_performance_comparison(self, metric_name: str, figsize: Tuple[int, int] = (15, 6)):
        """
        Create comprehensive visualization plots for performance comparison.
        
        Args:
            metric_name: Name of the metric to plot (must exist in results)
            figsize: Figure size as (width, height)
        """
        if metric_name not in self.results:
            raise ValueError(f"Metric '{metric_name}' not found. Run analyze_performance first.")
        
        results = self.results[metric_name]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{metric_name} Performance Comparison: {results.method1_name} vs {results.method2_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Box plots comparison
        ax1 = axes[0, 0]
        data_to_plot = [results.method1.data, results.method2.data]
        box_plot = ax1.boxplot(data_to_plot, 
                              labels=[results.method1_name, results.method2_name],
                              patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            
        ax1.set_title('Distribution Comparison (Box Plot)')
        ax1.set_ylabel(f'{metric_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histograms overlay
        ax2 = axes[0, 1]
        ax2.hist(results.method1.data, alpha=0.7, bins=max(5, len(results.method1.data)//2), 
                label=results.method1_name, color='lightblue', density=True)
        ax2.hist(results.method2.data, alpha=0.7, bins=max(5, len(results.method2.data)//2), 
                label=results.method2_name, color='lightcoral', density=True)
        ax2.set_title('Distribution Overlay (Histograms)')
        ax2.set_xlabel(f'{metric_name}')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Time series of stable epochs
        ax3 = axes[1, 0]
        epochs1 = range(self.stable_epoch_start, self.stable_epoch_start + len(results.method1.data))
        epochs2 = range(self.stable_epoch_start, self.stable_epoch_start + len(results.method2.data))
        ax3.plot(epochs1, results.method1.data, 'o-', alpha=0.7, 
                label=results.method1_name, color='blue', markersize=4)
        ax3.plot(epochs2, results.method2.data, 's-', alpha=0.7, 
                label=results.method2_name, color='red', markersize=4)
        ax3.set_title(f'Performance Over Stable Epochs (from epoch {self.stable_epoch_start})')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel(f'{metric_name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Percentile comparison bar chart
        ax4 = axes[1, 1]
        metrics = ['p50', 'p95', 'mean']
        method1_values = [results.method1.p50, results.method1.p95, results.method1.mean]
        method2_values = [results.method2.p50, results.method2.p95, results.method2.mean]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, method1_values, width, label=results.method1_name, 
                       color='lightblue', alpha=0.8)
        bars2 = ax4.bar(x + width/2, method2_values, width, label=results.method2_name, 
                       color='lightcoral', alpha=0.8)
        
        ax4.set_title('Key Metrics Comparison')
        ax4.set_ylabel(f'{metric_name}')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
    def print_summary(self, metric_name: str):
        """
        Print a comprehensive summary of the analysis results.
        
        Args:
            metric_name: Name of the metric to summarize
        """
        if metric_name not in self.results:
            raise ValueError(f"Metric '{metric_name}' not found. Run analyze_performance first.")
        
        results = self.results[metric_name]
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE ANALYSIS SUMMARY: {metric_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nAnalysis Configuration:")
        print(f"  • Stable epochs analyzed: {results.stable_epochs_analyzed} (from epoch {self.stable_epoch_start})")
        print(f"  • Methods compared: {results.method1_name} vs {results.method2_name}")
        
        print(f"\n{results.method1_name.upper()} Performance:")
        print(f"  • p50 (median): {results.method1.p50:.3f}")
        print(f"  • p95 (95th percentile): {results.method1.p95:.3f}")
        print(f"  • Mean: {results.method1.mean:.3f}")
        print(f"  • Standard Deviation: {results.method1.std:.3f}")
        
        print(f"\n{results.method2_name.upper()} Performance:")
        print(f"  • p50 (median): {results.method2.p50:.3f}")
        print(f"  • p95 (95th percentile): {results.method2.p95:.3f}")
        print(f"  • Mean: {results.method2.mean:.3f}")
        print(f"  • Standard Deviation: {results.method2.std:.3f}")
        
        print(f"\n🏆 WINNER ANALYSIS:")
        print(f"  • Best p50 (median): {results.comparison.better_p50} "
              f"({results.comparison.p50_improvement_percent:.1f}% better)")
        print(f"  • Best p95 (95th percentile): {results.comparison.better_p95} "
              f"({results.comparison.p95_improvement_percent:.1f}% better)")
        print(f"  • Best mean: {results.comparison.better_mean}")
        
        # Overall recommendation
        scores = {results.method1_name: 0, results.method2_name: 0}
        scores[results.comparison.better_p50] += 1
        scores[results.comparison.better_p95] += 1
        scores[results.comparison.better_mean] += 1
        
        winner = max(scores, key=scores.get)
        print(f"\n🎯 OVERALL RECOMMENDATION: {winner}")
        print(f"   Wins in {scores[winner]}/3 key metrics")
        
        if scores[winner] == 3:
            print(f"   {winner} is clearly superior across all metrics!")
        elif scores[winner] == 2:
            print(f"   {winner} shows better performance in most metrics.")
        else:
            print(f"   Performance is mixed - consider other factors for final decision.")