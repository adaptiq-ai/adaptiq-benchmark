import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from datetime import datetime

import numpy as np
from scipy import stats
import matplotlib.patches as patches
# --- Configuration ---
# Use the file paths from your provided JSON data
CONFIG = {
    "test_run_id": "20250813_100138",
    "llm_model": "GPT-4.1",
    "image_backend": "black-forest-labs/flux-1.1-pro",
    "data_dir": './test_results/20250813_100138',
    "data_notes": (
        "Notes on sample sizes:\n"
        "- Cost/Latency/Token data captured for 100 image pairs.\n"
        "- CLIP quality scores were successfully computed for 93 pairs due to\n  intermittent API errors or invalid image formats.\n"
        "- All per-pair comparisons are limited to the 93 pairs with complete data."
    ),
    "colors": {
        "Adaptiq": "#4C72B0", # Blue
        "GPT": "#55A868",     # Green
        "Savings": "#4C72B0", # Blue for savings
        "Loss": "#C44E52",      # Red for losses/worse performance
        "Tie": "#8C8C8C"      # Grey for ties
    }
}
METRICS_FILE = './test_results/20250813_100138/metrics.json'
PROMPTS_FILE = './test_results/20250813_100138/image_prompts.json'
OUTPUT_PDF = f'adaptiq_vs_gpt_multipage_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

# --- Helper Functions ---

def load_json_data(filepath):
    """Loads data from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return None

def create_performance_dataframe(prompts_data):
    """Creates a Pandas DataFrame from the image_prompts.json data."""
    records = []
    if prompts_data and 'results' in prompts_data:
        for item in prompts_data['results']:
            method = 'Adaptiq' if item['agent_prompt_type'] == 'adaptiq' else 'GPT'
            records.append({
                'method': method,
                'latency': item['execution_time'],
                'tokens': item['metrics']['total_tokens'],
                'total_cost': item['metrics']['total_cost']
            })
    return pd.DataFrame(records)

def create_title_page(pdf_pages):
    """Creates a title page."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, 'Adaptiq vs GPT', ha='center', va='center', 
            fontsize=32, fontweight='bold', transform=ax.transAxes)
    
    ax.text(0.5, 0.6, 'Comprehensive Performance Analysis', ha='center', va='center', 
            fontsize=24, transform=ax.transAxes)
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(0.5, 0.4, f'Generated on: {timestamp}', ha='center', va='center', 
            fontsize=16, style='italic', transform=ax.transAxes)
    
    # Add a decorative border
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_latency_plot(performance_df, pdf_pages):
    """Creates latency comparison plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    palette = {"Adaptiq": "#4C72B0", "GPT": "#55A868"}
    
    if not performance_df.empty:
        # Get sample sizes
        n_adaptiq = len(performance_df[performance_df['method'] == 'Adaptiq'])
        n_gpt = len(performance_df[performance_df['method'] == 'GPT'])
        
        sns.boxplot(x='method', y='latency', data=performance_df, palette=palette, width=0.6, ax=ax)
        sns.stripplot(x='method', y='latency', data=performance_df, color=".25", size=8, ax=ax)
        
        # ADDED: Sample size in title
        ax.set_title(f'Latency Comparison (N = {min(n_adaptiq, n_gpt)} pairs)', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Execution Time (Seconds)', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add statistics text with sample sizes
        stats_text = f"Sample Sizes: AdaptiQ={n_adaptiq}, GPT={n_gpt}\n\n"
        for method in ['Adaptiq', 'GPT']:
            method_data = performance_df[performance_df['method'] == method]['latency']
            if not method_data.empty:
                mean_val = method_data.mean()
                std_val = method_data.std()
                stats_text += f"{method}: {mean_val:.2f}±{std_val:.2f}s\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No latency data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_tokens_plot(performance_df, pdf_pages):
    """Creates token usage comparison plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    palette = {"Adaptiq": "#4C72B0", "GPT": "#55A868"}
    
    if not performance_df.empty:
        sns.boxplot(x='method', y='tokens', data=performance_df, palette=palette, width=0.6, ax=ax)
        sns.stripplot(x='method', y='tokens', data=performance_df, color=".25", size=8, ax=ax)
        
        ax.set_title('Token Usage Comparison', fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Total Tokens', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add statistics text
        stats_text = ""
        for method in ['Adaptiq', 'GPT']:
            method_data = performance_df[performance_df['method'] == method]['tokens']
            if not method_data.empty:
                mean_val = method_data.mean()
                std_val = method_data.std()
                stats_text += f"{method}: {mean_val:.0f}±{std_val:.0f} tokens\n"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No token data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_cost_plot(performance_df, pdf_pages):
    """Creates cost comparison plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    palette = {"Adaptiq": "#4C72B0", "GPT": "#55A868"}
    
    if not performance_df.empty:
        sns.boxplot(x='method', y='total_cost', data=performance_df, palette=palette, width=0.6, ax=ax)
        sns.stripplot(x='method', y='total_cost', data=performance_df, color=".25", size=8, ax=ax)
        
        ax.set_title('Cost Distribution per Image', fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Cost per Image ($)', fontsize=14)
        ax.set_xlabel('Method', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add statistics text and savings calculation
        stats_text = ""
        costs = {}
        for method in ['Adaptiq', 'GPT']:
            method_data = performance_df[performance_df['method'] == method]['total_cost']
            if not method_data.empty:
                mean_val = method_data.mean()
                std_val = method_data.std()
                costs[method] = mean_val
                stats_text += f"Avg {method}: ${mean_val:.4f}±${std_val:.4f}\n"
        
        if 'GPT' in costs and 'Adaptiq' in costs:
            savings = costs['GPT'] - costs['Adaptiq']
            savings_pct = (savings / costs['GPT'] * 100) if costs['GPT'] > 0 else 0
            stats_text += f"\nAvg Savings: ${savings:.4f} ({savings_pct:.1f}%)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_cumulative_cost_plot(performance_df, pdf_pages):
    """Creates a cumulative cost comparison plot over time."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if not performance_df.empty:
        # Separate data and calculate cumulative sum
        adaptiq_df = performance_df[performance_df['method'] == 'Adaptiq'].copy()
        gpt_df = performance_df[performance_df['method'] == 'GPT'].copy()
        
        if not adaptiq_df.empty and not gpt_df.empty:
            # Reset indices to ensure they start from 0
            adaptiq_df = adaptiq_df.reset_index(drop=True)
            gpt_df = gpt_df.reset_index(drop=True)
            
            adaptiq_df['cumulative_cost'] = adaptiq_df['total_cost'].cumsum()
            gpt_df['cumulative_cost'] = gpt_df['total_cost'].cumsum()
            
            # Add a run number for the x-axis
            adaptiq_df['run'] = range(1, len(adaptiq_df) + 1)
            gpt_df['run'] = range(1, len(gpt_df) + 1)
            
            # Ensure both series have the same length for comparison
            min_length = min(len(adaptiq_df), len(gpt_df))
            adaptiq_subset = adaptiq_df.iloc[:min_length]
            gpt_subset = gpt_df.iloc[:min_length]
            
            # Plotting
            ax.plot(adaptiq_subset['run'], adaptiq_subset['cumulative_cost'], 
                   label='Adaptiq', color='#4C72B0', marker='o', linestyle='-')
            ax.plot(gpt_subset['run'], gpt_subset['cumulative_cost'], 
                   label='GPT', color='#55A868', marker='o', linestyle='-')
            
            # Shade the area between the lines to show savings
            runs = adaptiq_subset['run'].values
            adaptiq_costs = adaptiq_subset['cumulative_cost'].values
            gpt_costs = gpt_subset['cumulative_cost'].values
            
            ax.fill_between(runs, adaptiq_costs, gpt_costs, 
                            where=(gpt_costs >= adaptiq_costs), 
                            facecolor='#4C72B0', alpha=0.3, interpolate=True, label='Adaptiq Savings')
            ax.fill_between(runs, adaptiq_costs, gpt_costs, 
                            where=(gpt_costs < adaptiq_costs), 
                            facecolor='#C44E52', alpha=0.3, interpolate=True, label='GPT Savings')
            
            # FIXED: Correct title with actual sample size
            ax.set_title(f'Cumulative Cost Over Time (N = {min_length} pairs)', 
                        fontsize=20, fontweight='bold', pad=20)
            ax.set_xlabel('Image Generation Number', fontsize=14)
            ax.set_ylabel('Cumulative Cost ($)', fontsize=14)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Add summary text with corrected delta
            total_adaptiq_cost = adaptiq_subset['cumulative_cost'].iloc[-1]
            total_gpt_cost = gpt_subset['cumulative_cost'].iloc[-1]
            total_savings = total_gpt_cost - total_adaptiq_cost  # GPT cost - AdaptiQ cost
            
            summary_text = f"Cumulative Cost (N = {min_length} pairs):\n"
            summary_text += f"  AdaptiQ: ${total_adaptiq_cost:.4f}\n"
            summary_text += f"  GPT: ${total_gpt_cost:.4f}\n\n"
            summary_text += f"Total Savings (GPT - AdaptiQ): ${total_savings:.4f}"
            
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_efficiency_score_plot(performance_df, metrics_data, pdf_pages):
    """Creates an efficiency score plot combining cost, speed, and quality."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if not performance_df.empty and metrics_data and 'clip' in metrics_data:
        try:
            # Calculate efficiency scores for each method
            efficiency_data = []
            
            for method in ['Adaptiq', 'GPT']:
                method_df = performance_df[performance_df['method'] == method]
                if not method_df.empty:
                    # Normalize metrics (lower is better for cost/latency, higher for quality)
                    avg_cost = method_df['total_cost'].mean()
                    avg_latency = method_df['latency'].mean()
                    
                    # Get quality scores
                    if method == 'Adaptiq':
                        quality_scores = [item['adaptiq_image_score']['scaled_score'] for item in metrics_data['clip']]
                    else:
                        quality_scores = [item['gpt_image_score']['scaled_score'] for item in metrics_data['clip']]
                    
                    avg_quality = np.mean(quality_scores)
                    
                    # Calculate efficiency score (quality per dollar per second)
                    efficiency_score = avg_quality / (avg_cost * avg_latency) if avg_cost > 0 and avg_latency > 0 else 0
                    
                    efficiency_data.append({
                        'Method': method,
                        'Efficiency Score': efficiency_score,
                        'Avg Cost': avg_cost,
                        'Avg Latency': avg_latency,
                        'Avg Quality': avg_quality
                    })
            
            if efficiency_data:
                methods = [d['Method'] for d in efficiency_data]
                scores = [d['Efficiency Score'] for d in efficiency_data]
                colors = ['#4C72B0', '#55A868']
                
                bars = ax.bar(methods, scores, color=colors)
                ax.set_title('Overall Efficiency Score\n(Quality per Dollar per Second)', 
                           fontsize=20, fontweight='bold', pad=20)
                ax.set_ylabel('Efficiency Score', fontsize=14)
                ax.set_xlabel('Method', fontsize=14)
                
                # Add value labels on bars
                for bar, data in zip(bars, efficiency_data):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Add breakdown text
                breakdown_text = "Breakdown:\n"
                for data in efficiency_data:
                    breakdown_text += f"{data['Method']}:\n"
                    breakdown_text += f"  Quality: {data['Avg Quality']:.3f}\n"
                    breakdown_text += f"  Cost: ${data['Avg Cost']:.4f}\n"
                    breakdown_text += f"  Latency: {data['Avg Latency']:.2f}s\n\n"
                
                ax.text(0.98, 0.98, breakdown_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        except Exception as e:
            ax.text(0.5, 0.5, f'Efficiency calculation failed\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for efficiency analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_performance_radar_chart(performance_df, metrics_data, pdf_pages):
    """Creates a radar chart comparing multiple performance dimensions."""
    fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw=dict(projection='polar'))
    
    if not performance_df.empty and metrics_data and 'clip' in metrics_data:
        try:
            categories = ['Speed\n(1/Latency)', 'Cost Efficiency\n(1/Cost)', 'Image Quality', 
                         'Token Efficiency\n(Quality/Tokens)', 'Consistency\n(1/StdDev)']
            
            radar_data = {}
            
            for method in ['Adaptiq', 'GPT']:
                method_df = performance_df[performance_df['method'] == method]
                if not method_df.empty:
                    # Get quality scores
                    if method == 'Adaptiq':
                        quality_scores = [item['adaptiq_image_score']['scaled_score'] for item in metrics_data['clip']]
                    else:
                        quality_scores = [item['gpt_image_score']['scaled_score'] for item in metrics_data['clip']]
                    
                    # Calculate normalized metrics (0-1 scale, higher is better)
                    speed_score = 1 / method_df['latency'].mean() if method_df['latency'].mean() > 0 else 0
                    cost_efficiency = 1 / method_df['total_cost'].mean() if method_df['total_cost'].mean() > 0 else 0
                    quality_score = np.mean(quality_scores)
                    token_efficiency = quality_score / method_df['tokens'].mean() if method_df['tokens'].mean() > 0 else 0
                    consistency = 1 / (np.std(quality_scores) + 0.001)  # Add small value to avoid division by zero
                    
                    radar_data[method] = [speed_score, cost_efficiency, quality_score, token_efficiency, consistency]
            
            # Normalize all values to 0-1 scale across methods
            if len(radar_data) == 2:
                all_values = np.array(list(radar_data.values()))
                max_values = np.max(all_values, axis=0)
                
                for method in radar_data:
                    radar_data[method] = [val/max_val if max_val > 0 else 0 
                                        for val, max_val in zip(radar_data[method], max_values)]
            
            # Plot radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = {'Adaptiq': '#4C72B0', 'GPT': '#55A868'}
            
            for method, values in radar_data.items():
                values += values[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[method])
                ax.fill(angles, values, alpha=0.25, color=colors[method])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_title('Performance Radar Chart\n(Normalized Scores)', fontsize=20, fontweight='bold', pad=30)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
            ax.grid(True)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Radar chart generation failed\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for radar chart', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_correlation_heatmap(performance_df, metrics_data, pdf_pages):
    """Creates a correlation heatmap between different metrics."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if not performance_df.empty and metrics_data and 'clip' in metrics_data:
        try:
            # Create correlation dataframe
            corr_data = []
            
            for i, item in enumerate(metrics_data['clip']):
                if i < len(performance_df) // 2:  # Assuming alternating Adaptiq/GPT entries
                    adaptiq_row = performance_df[performance_df['method'] == 'Adaptiq'].iloc[i]
                    gpt_row = performance_df[performance_df['method'] == 'GPT'].iloc[i]
                    
                    corr_data.append({
                        'Adaptiq_Cost': adaptiq_row['total_cost'],
                        'GPT_Cost': gpt_row['total_cost'],
                        'Adaptiq_Latency': adaptiq_row['latency'],
                        'GPT_Latency': gpt_row['latency'],
                        'Adaptiq_Tokens': adaptiq_row['tokens'],
                        'GPT_Tokens': gpt_row['tokens'],
                        'Adaptiq_Quality': item['adaptiq_image_score']['scaled_score'],
                        'GPT_Quality': item['gpt_image_score']['scaled_score'],
                        'Quality_Difference': item['gpt_image_score']['scaled_score'] - item['adaptiq_image_score']['scaled_score']
                    })
            
            if corr_data:
                corr_df = pd.DataFrame(corr_data)
                correlation_matrix = corr_df.corr()
                
                # Create heatmap
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
                
                ax.set_title('Correlation Matrix of Performance Metrics', fontsize=20, fontweight='bold', pad=20)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
            else:
                ax.text(0.5, 0.5, 'Insufficient data for correlation analysis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=16)
        except Exception as e:
            ax.text(0.5, 0.5, f'Correlation analysis failed\nError: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
    else:
        ax.text(0.5, 0.5, 'No data available for correlation analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_distribution_comparison_plot(performance_df, pdf_pages):
    """Creates distribution comparison plots for all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Distribution Comparison: Adaptiq vs GPT', fontsize=20, fontweight='bold')
    
    metrics = ['latency', 'tokens', 'total_cost']
    titles = ['Latency Distribution (s)', 'Token Usage Distribution', 'Cost Distribution ($)']
    
    if not performance_df.empty:
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if i < 3:
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                
                adaptiq_data = performance_df[performance_df['method'] == 'Adaptiq'][metric]
                gpt_data = performance_df[performance_df['method'] == 'GPT'][metric]
                
                if not adaptiq_data.empty and not gpt_data.empty:
                    ax.hist([adaptiq_data, gpt_data], bins=10, alpha=0.7, 
                           label=['Adaptiq', 'GPT'], color=['#4C72B0', '#55A868'])
                    ax.set_title(title, fontsize=14, fontweight='bold')
                    ax.set_xlabel(metric.replace('_', ' ').title())
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(alpha=0.3)
                else:
                    ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
        
        # Statistical comparison in the last subplot
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('Statistical Tests', fontsize=14, fontweight='bold')
        
        stats_text = "Mann-Whitney U Test Results:\n\n"
        for metric in metrics:
            adaptiq_data = performance_df[performance_df['method'] == 'Adaptiq'][metric]
            gpt_data = performance_df[performance_df['method'] == 'GPT'][metric]
            
            if len(adaptiq_data) > 1 and len(gpt_data) > 1:
                try:
                    statistic, p_value = stats.mannwhitneyu(adaptiq_data, gpt_data, alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    stats_text += f"{metric}: p={p_value:.4f} {significance}\n"
                except:
                    stats_text += f"{metric}: Test failed\n"
        
        stats_text += "\n* p<0.05, ** p<0.01, *** p<0.001\nns = not significant"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_time_series_analysis(performance_df, pdf_pages):
    """Creates a time series analysis showing performance trends over sequential runs."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Performance Trends Over Sequential Runs', fontsize=20, fontweight='bold')
    
    if not performance_df.empty:
        # Add run numbers
        df_copy = performance_df.copy()
        df_copy['run_id'] = df_copy.index
        
        metrics = ['latency', 'tokens', 'total_cost']
        titles = ['Latency Trend', 'Token Usage Trend', 'Cost Trend']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if i < 3:
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                
                for method in ['Adaptiq', 'GPT']:
                    method_data = df_copy[df_copy['method'] == method]
                    if not method_data.empty:
                        ax.plot(method_data['run_id'], method_data[metric], 
                               marker='o', label=method, linewidth=2,
                               color='#4C72B0' if method == 'Adaptiq' else '#55A868')
                
                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_xlabel('Run Number')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.legend()
                ax.grid(alpha=0.3)
        
        # Moving averages in the last subplot
        ax = axes[1, 1]
        window_size = min(3, len(df_copy) // 4) if len(df_copy) >= 4 else 1
        
        if window_size > 1:
            for method in ['Adaptiq', 'GPT']:
                method_data = df_copy[df_copy['method'] == method].copy()
                if len(method_data) >= window_size:
                    method_data['cost_ma'] = method_data['total_cost'].rolling(window=window_size, center=True).mean()
                    ax.plot(method_data['run_id'], method_data['cost_ma'], 
                           marker='s', label=f'{method} (MA-{window_size})', linewidth=2,
                           color='#4C72B0' if method == 'Adaptiq' else '#55A868')
            
            ax.set_title(f'Cost Moving Average (Window={window_size})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Cost ($)')
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for moving average', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_roi_analysis_plot(performance_df, metrics_data, pdf_pages):
    """Creates an ROI analysis showing value for money."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle('Return on Investment (ROI) Analysis', fontsize=20, fontweight='bold')
    
    if not performance_df.empty and metrics_data and 'clip' in metrics_data:
        try:
            # Calculate ROI for each image
            roi_data = []
            
            adaptiq_costs = performance_df[performance_df['method'] == 'Adaptiq']['total_cost'].tolist()
            gpt_costs = performance_df[performance_df['method'] == 'GPT']['total_cost'].tolist()
            
            for i, item in enumerate(metrics_data['clip']):
                if i < len(adaptiq_costs) and i < len(gpt_costs):
                    adaptiq_roi = item['adaptiq_image_score']['scaled_score'] / adaptiq_costs[i] if adaptiq_costs[i] > 0 else 0
                    gpt_roi = item['gpt_image_score']['scaled_score'] / gpt_costs[i] if gpt_costs[i] > 0 else 0
                    
                    roi_data.append({
                        'Image': f'Img_{i+1}',
                        'Adaptiq_ROI': adaptiq_roi,
                        'GPT_ROI': gpt_roi,
                        'ROI_Advantage': adaptiq_roi - gpt_roi
                    })
            
            if roi_data:
                roi_df = pd.DataFrame(roi_data)
                
                # ROI comparison
                x = np.arange(len(roi_df))
                width = 0.35
                
                ax1.bar(x - width/2, roi_df['Adaptiq_ROI'], width, label='Adaptiq', color='#4C72B0')
                ax1.bar(x + width/2, roi_df['GPT_ROI'], width, label='GPT', color='#55A868')
                
                ax1.set_title('Quality ROI per Image\n(Quality Score / Cost)', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Images')
                ax1.set_ylabel('ROI (Quality per Dollar)')
                ax1.set_xticks(x)
                ax1.set_xticklabels(roi_df['Image'], rotation=45)
                ax1.legend()
                ax1.grid(alpha=0.3)
                
                # ROI advantage
                colors = ['#4C72B0' if x > 0 else '#C44E52' for x in roi_df['ROI_Advantage']]
                ax2.bar(range(len(roi_df)), roi_df['ROI_Advantage'], color=colors)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title('Adaptiq ROI Advantage\n(Positive = Better Value)', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Images')
                ax2.set_ylabel('ROI Difference')
                ax2.set_xticks(range(len(roi_df)))
                ax2.set_xticklabels(roi_df['Image'], rotation=45)
                ax2.grid(alpha=0.3)
                
                # Add summary statistics
                avg_adaptiq_roi = roi_df['Adaptiq_ROI'].mean()
                avg_gpt_roi = roi_df['GPT_ROI'].mean()
                advantage_count = (roi_df['ROI_Advantage'] > 0).sum()
                
                summary_text = f"Average ROI:\n"
                summary_text += f"Adaptiq: {avg_adaptiq_roi:.2f}\n"
                summary_text += f"GPT: {avg_gpt_roi:.2f}\n\n"
                summary_text += f"Adaptiq advantage: {advantage_count}/{len(roi_df)} images"
                
                fig.text(0.02, 0.02, summary_text, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'ROI analysis failed\nError: {str(e)}', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax2.axis('off')
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_clip_winners_plot(metrics_data, pdf_pages):
    """Creates CLIP score winners plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    palette = {"Adaptiq": "#4C72B0", "GPT": "#55A868", "Tie": "#C44E52"}
    
    if metrics_data and 'clip' in metrics_data:
        winners = [item['winner'] for item in metrics_data['clip']]
        winner_counts = pd.Series(winners).value_counts()
        order = ['Adaptiq', 'GPT', 'Tie']
        winner_counts = winner_counts.reindex(order).fillna(0)
        
        bars = ax.bar(winner_counts.index, winner_counts.values, 
                     color=[palette.get(i, '#C44E52') for i in winner_counts.index])
        
        ax.set_title('Image Quality Winners (CLIP Scores)', fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Number of Images', fontsize=14)
        ax.set_xlabel('Winner', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), 
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        total_images = len(winners)
        adaptiq_wins = winners.count('Adaptiq')
        gpt_wins = winners.count('GPT')
        ties = winners.count('Tie')
        
        summary_text = f"Total Images: {total_images}\n"
        summary_text += f"Adaptiq Wins: {adaptiq_wins} ({adaptiq_wins/total_images*100:.1f}%)\n"
        summary_text += f"GPT Wins: {gpt_wins} ({gpt_wins/total_images*100:.1f}%)\n"
        summary_text += f"Ties: {ties} ({ties/total_images*100:.1f}%)"
        
        ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No CLIP score data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_clip_scores_plot(metrics_data, pdf_pages):
    """Creates detailed CLIP scores plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if metrics_data and 'clip' in metrics_data:
        clip_data = []
        max_images = len(metrics_data['clip'])
        
        for i, item in enumerate(metrics_data['clip']):
            img_name = f"Img_{i+1}"
            clip_data.append({
                'Image': img_name,
                'Adaptiq': item['adaptiq_image_score']['scaled_score'],
                'GPT': item['gpt_image_score']['scaled_score']
            })
        
        clip_df = pd.DataFrame(clip_data)
        clip_df.set_index('Image').plot(kind='bar', ax=ax, color=["#4C72B0", "#55A868"], 
                                       width=0.8, figsize=(11, 8.5))
        
        ax.set_title(f'CLIP Similarity Scores (All {max_images} Images)', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_ylabel('Similarity Score', fontsize=14)
        ax.set_xlabel('Images', fontsize=14)
        ax.tick_params(axis='x', labelsize=10, rotation=90)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        adaptiq_avg = clip_df['Adaptiq'].mean()
        gpt_avg = clip_df['GPT'].mean()
        ax.axhline(adaptiq_avg, color='#4C72B0', linestyle='--', alpha=0.7, linewidth=2)
        ax.axhline(gpt_avg, color='#55A868', linestyle='--', alpha=0.7, linewidth=2)
        
        avg_text = f"Avg Adaptiq: {adaptiq_avg:.3f}\nAvg GPT: {gpt_avg:.3f}"
        ax.text(0.02, 0.98, avg_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No CLIP score details available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_cost_savings_plot(performance_df, prompts_data, pdf_pages):
    """Creates cost savings per image plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if not performance_df.empty and prompts_data:
        try:
            df_copy = performance_df.copy()
            df_copy['pair_id'] = df_copy.index // 2
            
            cost_pivot = df_copy.pivot(index='pair_id', columns='method', values='total_cost').reset_index()
            
            if 'GPT' in cost_pivot.columns and 'Adaptiq' in cost_pivot.columns:
                cost_pivot['margin'] = cost_pivot['GPT'] - cost_pivot['Adaptiq']
                
                image_names = [f"Img_{i+1}" for i in range(len(cost_pivot))]
                cost_pivot['image_name'] = image_names
                
                bar_colors = ['#4C72B0' if margin > 0 else '#C44E52' for margin in cost_pivot['margin']]
                
                ax.bar(range(len(cost_pivot)), cost_pivot['margin'], color=bar_colors)
                ax.set_title('Cost Savings per Image (GPT Cost - Adaptiq Cost)', 
                           fontsize=20, fontweight='bold', pad=20)
                ax.set_ylabel('Cost Difference ($)', fontsize=14)
                ax.set_xlabel('Images', fontsize=14)
                ax.set_xticks(range(len(cost_pivot)))
                ax.set_xticklabels(cost_pivot['image_name'], rotation=90, fontsize=10)
                ax.tick_params(axis='y', labelsize=12)
                ax.grid(axis='y', alpha=0.3)
                
                avg_savings = cost_pivot['margin'].mean()
                ax.axhline(avg_savings, color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                positive_savings = (cost_pivot['margin'] > 0).sum()
                total_images = len(cost_pivot)
                total_savings = cost_pivot['margin'].sum()
                
                stats_text = f"Images where Adaptiq saved money: {positive_savings}/{total_images}\n"
                stats_text += f"Average savings per image: ${avg_savings:.4f}\n"
                stats_text += f"Total savings across all images: ${total_savings:.4f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
                
                legend_elements = [mpatches.Patch(color='#4C72B0', label='Adaptiq Cheaper'),
                                 mpatches.Patch(color='#C44E52', label='GPT Cheaper'),
                                 mpatches.Patch(color='red', label=f'Average Savings: ${avg_savings:.4f}')]
                ax.legend(handles=legend_elements, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'Insufficient cost data for comparison', ha='center', va='center', 
                        transform=ax.transAxes, fontsize=16)
        except Exception as e:
            ax.text(0.5, 0.5, f'Cost analysis unavailable\nError: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
    else:
        ax.text(0.5, 0.5, 'No cost data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_quality_vs_cost_plot(metrics_data, performance_df, pdf_pages):
    """Creates a scatter plot of image quality vs. cost."""
    fig, ax = plt.subplots(figsize=(11, 8.5))

    if metrics_data and 'clip' in metrics_data and not performance_df.empty:
        try:
            plot_data = []
            adaptiq_costs = performance_df[performance_df['method'] == 'Adaptiq']['total_cost'].tolist()
            gpt_costs = performance_df[performance_df['method'] == 'GPT']['total_cost'].tolist()
            
            for i, item in enumerate(metrics_data['clip']):
                if i < len(adaptiq_costs):
                    plot_data.append({
                        'method': 'Adaptiq',
                        'cost': adaptiq_costs[i],
                        'quality': item['adaptiq_image_score']['scaled_score'],
                        'image': f"Img_{i+1}"
                    })
                if i < len(gpt_costs):
                    plot_data.append({
                        'method': 'GPT',
                        'cost': gpt_costs[i],
                        'quality': item['gpt_image_score']['scaled_score'],
                        'image': f"Img_{i+1}"
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            palette = {"Adaptiq": "#4C72B0", "GPT": "#55A868"}
            sns.scatterplot(data=plot_df, x='cost', y='quality', hue='method', palette=palette, s=100, alpha=0.8, ax=ax)
            
            ax.set_title('Image Quality vs. Generation Cost', fontsize=20, fontweight='bold', pad=20)
            ax.set_xlabel('Cost per Image ($)', fontsize=14)
            ax.set_ylabel('Image Quality (CLIP Score)', fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(title='Method', fontsize=12)
            
            # Add ideal quadrant labels for interpretation
            mean_cost = plot_df['cost'].mean()
            mean_quality = plot_df['quality'].mean()
            ax.axvline(mean_cost, color='grey', linestyle='--', alpha=0.5)
            ax.axhline(mean_quality, color='grey', linestyle='--', alpha=0.5)
            
            ax.text(0.02, 0.98, 'Pricier & Better', transform=ax.transAxes, ha='left', va='top', fontsize=10, color='orange', alpha=0.7)
            ax.text(0.98, 0.98, 'Cheaper & Better\n(Ideal)', transform=ax.transAxes, ha='right', va='top', fontsize=10, color='green', alpha=0.7, fontweight='bold')
            ax.text(0.02, 0.02, 'Pricier & Worse\n(Worst)', transform=ax.transAxes, ha='left', va='bottom', fontsize=10, color='red', alpha=0.7, fontweight='bold')
            ax.text(0.98, 0.02, 'Cheaper & Worse', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='blue', alpha=0.7)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Plot generation failed.\nError: {str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
    else:
        ax.text(0.5, 0.5, 'Insufficient data for Quality vs. Cost plot', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)

    plt.tight_layout()
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_prompt_quality_plot(metrics_data, pdf_pages):
    """Creates prompt quality comparison plot."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    
    if metrics_data and 'prompt_quality_score' in metrics_data:
        pq_data = metrics_data['prompt_quality_score']
        scores = pq_data['overall_scores']
        methods = list(scores.keys())
        values = list(scores.values())
        
        bars = ax.barh(methods, values, color=['#4C72B0', '#55A868'])
        ax.set_title('Prompt Engineering Quality Scores', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Quality Score (0-100)', fontsize=14)
        ax.set_ylabel('Method', fontsize=14)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(axis='x', alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width - 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}', 
                   ha='right', va='center', color='white', fontsize=14, fontweight='bold')
        
        ax.invert_yaxis()
        
        if 'summary' in pq_data:
            summary = pq_data['summary']
            fig.text(0.1, 0.1, f"Summary:\n{summary}", va='top', ha='left',
                     wrap=True, fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No prompt quality data available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=16)
    
    plt.tight_layout(rect=[0.1, 0.2, 0.9, 0.9])
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_summary_page(metrics_data, performance_df, pdf_pages):
    """Creates a summary statistics page."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle('Performance Summary & Key Insights', fontsize=20, fontweight='bold')
    
    # Summary statistics table
    ax1.axis('off')
    ax1.set_title('Performance Statistics', fontsize=16, fontweight='bold', pad=10)
    
    if not performance_df.empty:
        summary_stats = performance_df.groupby('method').agg({
            'latency': ['mean', 'std'],
            'tokens': ['mean', 'std'], 
            'total_cost': ['mean', 'sum']
        }).round(4)
        
        summary_text = ""
        for method in ['Adaptiq', 'GPT']:
            if method in summary_stats.index:
                stats = summary_stats.loc[method]
                summary_text += f"{method}:\n"
                summary_text += f"  Avg Latency: {stats[('latency', 'mean')]:.2f}±{stats[('latency', 'std')]:.2f}s\n"
                summary_text += f"  Avg Tokens: {stats[('tokens', 'mean')]:.0f}±{stats[('tokens', 'std')]:.0f}\n"
                summary_text += f"  Avg Cost: ${stats[('total_cost', 'mean')]:.4f}\n"
                summary_text += f"  Total Cost: ${stats[('total_cost', 'sum')]:.4f}\n\n"
        
        ax1.text(0.01, 0.95, summary_text, transform=ax1.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    
    # Key metrics comparison
    ax2.axis('off')
    ax2.set_title('Key Metrics Comparison', fontsize=16, fontweight='bold', pad=10)
    
    if not performance_df.empty:
        adaptiq_data = performance_df[performance_df['method'] == 'Adaptiq']
        gpt_data = performance_df[performance_df['method'] == 'GPT']
        
        comparison_text = "ADAPTIQ vs GPT (Δ = AdaptiQ - GPT):\n\n"
        
        if not adaptiq_data.empty and not gpt_data.empty:
            # FIXED: AdaptiQ - GPT (positive means AdaptiQ is better for speed, worse for cost)
            latency_diff = adaptiq_data['latency'].mean() - gpt_data['latency'].mean()
            latency_pct = (latency_diff / gpt_data['latency'].mean() * 100) if gpt_data['latency'].mean() > 0 else 0
            comparison_text += f"Latency: {latency_diff:+.2f}s ({latency_pct:+.1f}%)\n"
            
            cost_diff = adaptiq_data['total_cost'].mean() - gpt_data['total_cost'].mean()
            cost_pct = (cost_diff / gpt_data['total_cost'].mean() * 100) if gpt_data['total_cost'].mean() > 0 else 0
            comparison_text += f"Cost: ${cost_diff:+.4f} ({cost_pct:+.1f}%)\n"
            
            token_diff = adaptiq_data['tokens'].mean() - gpt_data['tokens'].mean()
            token_pct = (token_diff / gpt_data['tokens'].mean() * 100) if gpt_data['tokens'].mean() > 0 else 0
            comparison_text += f"Tokens: {token_diff:+.0f} ({token_pct:+.1f}%)\n\n"
            
            # Add interpretation
            comparison_text += "Interpretation:\n"
            comparison_text += "Negative cost/latency = AdaptiQ better\n"
            comparison_text += "Positive quality = AdaptiQ better"
        
        ax2.text(0.01, 0.95, comparison_text, transform=ax2.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", alpha=0.8))
    
    # Image quality summary
    ax3.axis('off')
    ax3.set_title('Image Quality Summary', fontsize=16, fontweight='bold', pad=10)
    
    if metrics_data and 'clip' in metrics_data:
        winners = [item['winner'] for item in metrics_data['clip']]
        total_images = len(winners)
        adaptiq_wins = winners.count('AdaptiQ')
        gpt_wins = winners.count('GPT')
        ties = winners.count('Tie')
        
        adaptiq_pct = (adaptiq_wins / total_images * 100) if total_images > 0 else 0
        gpt_pct = (gpt_wins / total_images * 100) if total_images > 0 else 0
        ties_pct = (ties / total_images * 100) if total_images > 0 else 0

        clip_summary = f"CLIP Score Winners ({total_images} images):\n"
        clip_summary += f"  Adaptiq Wins: {adaptiq_wins} ({adaptiq_pct:.1f}%)\n"
        clip_summary += f"  GPT Wins: {gpt_wins} ({gpt_pct:.1f}%)\n"
        clip_summary += f"  Ties: {ties} ({ties_pct:.1f}%)\n\n"
        
        adaptiq_scores = [item['adaptiq_image_score']['scaled_score'] for item in metrics_data['clip']]
        gpt_scores = [item['gpt_image_score']['scaled_score'] for item in metrics_data['clip']]
        
        clip_summary += "Average CLIP Scores:\n"
        clip_summary += f"  Adaptiq: { (sum(adaptiq_scores) / len(adaptiq_scores)) if adaptiq_scores else 0:.3f}\n"
        clip_summary += f"  GPT: { (sum(gpt_scores) / len(gpt_scores)) if gpt_scores else 0:.3f}"

        
        ax3.text(0.01, 0.95, clip_summary, transform=ax3.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f8f0", alpha=0.8))
    
    # Overall conclusions
    ax4.axis('off')
    ax4.set_title('Key Insights', fontsize=16, fontweight='bold', pad=10)
    
    insights_text = "KEY FINDINGS:\n\n"
    if not performance_df.empty:
        adaptiq_cost = performance_df[performance_df['method'] == 'Adaptiq']['total_cost'].sum()
        gpt_cost = performance_df[performance_df['method'] == 'GPT']['total_cost'].sum()
        
        if not pd.isna(adaptiq_cost) and not pd.isna(gpt_cost):
            if adaptiq_cost < gpt_cost:
                savings_pct = ((gpt_cost - adaptiq_cost) / gpt_cost * 100)
                insights_text += f"💰 Adaptiq is {savings_pct:.1f}% more cost-effective.\n\n"
            else:
                extra_cost = ((adaptiq_cost - gpt_cost) / gpt_cost * 100)
                insights_text += f"💰 Adaptiq costs {extra_cost:.1f}% more.\n\n"
    
    if metrics_data and 'clip' in metrics_data:
        winners = [item['winner'] for item in metrics_data['clip']]
        adaptiq_wins = winners.count('Adaptiq')
        gpt_wins = winners.count('GPT')
        
        if adaptiq_wins > gpt_wins:
            insights_text += f"🎯 Adaptiq produces higher quality images more often.\n\n"
        elif gpt_wins > adaptiq_wins:
            insights_text += f"🎯 GPT produces higher quality images more often.\n\n"
        else:
            insights_text += f"🎯 Both methods have comparable image quality wins.\n\n"
    
    insights_text += "📊 See individual pages for detailed analysis."
    ax4.text(0.01, 0.95, insights_text, transform=ax4.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8dc", alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def create_setup_page(pdf_pages, config, prompts_data, metrics_data):
    """Creates a page detailing the benchmark configuration and sample sizes."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0.5, 0.85, 'Benchmark Configuration', ha='center', va='center',
            fontsize=24, fontweight='bold', transform=ax.transAxes)
    text_kwargs = {'fontsize': 14, 'fontfamily': 'monospace', 'va': 'top', 'ha': 'left'}

    # FIXED: Correct sample size calculation
    n_total_results = len(prompts_data['results']) if prompts_data and 'results' in prompts_data else 0
    n_pairs = n_total_results // 2  # Each pair has 2 results (AdaptiQ + GPT)
    n_clip = len(metrics_data['clip']) if metrics_data and 'clip' in metrics_data else 0

    setup_text = (
        f"BENCHMARK CONFIGURATION\n"
        f"{'='*50}\n\n"
        f"LLM/VLM Model : {config.get('llm_model', 'Not Specified')}\n"
        f"Image Model   : {config.get('image_backend', 'Not Specified')}\n"
        f"Test Run ID   : {config.get('test_run_id', 'N/A')}\n"
        f"Generated     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"SAMPLE SIZES\n"
        f"{'='*50}\n"
        f"Total Results      : {n_total_results} (AdaptiQ + GPT pairs)\n"
        f"Image Pairs        : {n_pairs} pairs\n"
        f"Cost/Latency Data  : N = {n_pairs} pairs ({n_total_results} individual results)\n"
        f"CLIP Quality Data  : N = {n_clip} pairs\n\n"
        f"DATA QUALITY NOTES\n"
        f"{'='*50}\n"
        f"Missing CLIP scores: {n_pairs - n_clip} pairs\n"
        f"Reason: API errors or invalid image formats\n\n"
        f"{config.get('data_notes', 'N/A')}"
    )
    ax.text(0.1, 0.7, setup_text, transform=ax.transAxes, **text_kwargs,
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", alpha=0.9))
    pdf_pages.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_multipage_pdf_report(metrics_data, prompts_data, output_file):
    """Creates a comprehensive multi-page PDF report with one plot per page."""
    
    print("Creating multi-page PDF report...")
    
    performance_df = create_performance_dataframe(prompts_data)
    
    with PdfPages(output_file) as pdf_pages:
        print("  📄 Creating title page...")
        create_title_page(pdf_pages)
        create_setup_page(pdf_pages, CONFIG, prompts_data, metrics_data)
        
        print("  📄 Creating summary page...")
        create_summary_page(metrics_data, performance_df, pdf_pages)
        
        print("  📊 Creating latency comparison plot...")
        create_latency_plot(performance_df, pdf_pages)
        
        print("  📊 Creating token usage plot...")
        create_tokens_plot(performance_df, pdf_pages)
        
        print("  📊 Creating cost per image plot...")
        create_cost_plot(performance_df, pdf_pages)
        
        print("  📈 Creating cumulative cost plot...")
        create_cumulative_cost_plot(performance_df, pdf_pages)
        
        print("  📊 Creating cost savings per image plot...")
        create_cost_savings_plot(performance_df, prompts_data, pdf_pages)

        create_efficiency_score_plot(performance_df, metrics_data, pdf_pages)
        create_performance_radar_chart(performance_df, metrics_data, pdf_pages)
        create_correlation_heatmap(performance_df, metrics_data, pdf_pages)
        create_distribution_comparison_plot(performance_df, pdf_pages)
        create_time_series_analysis(performance_df, pdf_pages)
        

        # print("  🏆 Creating CLIP winners plot...")
        # create_clip_winners_plot(metrics_data, pdf_pages)

        
        print("  📊 Creating detailed CLIP scores plot...")
        create_clip_scores_plot(metrics_data, pdf_pages)
        
        print("  ✨ Creating quality vs. cost scatter plot...")
        create_quality_vs_cost_plot(metrics_data, performance_df, pdf_pages)

        print("  📝 Creating prompt quality plot...")
        create_prompt_quality_plot(metrics_data, pdf_pages)
    
    return output_file

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading data and generating multi-page PDF report...")
    print("="*60)
    
    # Load from the provided strings
    try:
        # Load metrics.json from the multiline string
        metrics_json_string = open(METRICS_FILE, 'r',  encoding='utf-8').read()
        metrics_data = json.loads(metrics_json_string)

        # Load image_prompts.json from the multiline string
        prompts_json_string = open(PROMPTS_FILE, 'r',  encoding='utf-8').read()
        prompts_data = json.loads(prompts_json_string)
        
    except Exception as e:
        print(f"\n❌ Error loading JSON data: {e}")
        metrics_data, prompts_data = None, None

    if not metrics_data or not prompts_data:
        print("\n❌ Could not proceed with analysis due to missing or invalid data.")
    else:
        print("✅ Data loaded successfully!")
        print("\n🔄 Generating multi-page PDF report...")
        print("="*60)
        
        report_file = create_multipage_pdf_report(metrics_data, prompts_data, OUTPUT_PDF)
        
        print("\n" + "="*80)
        print("📋 MULTI-PAGE PDF REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"📄 Report saved as: {report_file}")
        page_count = 11
        print(f"📊 Report contains {page_count} pages:")
        print("   Page 1: Title Page")
        print("   Page 2: Summary & Key Insights")
        print("   Page 3: Latency Comparison")
        print("   Page 4: Token Usage Analysis")
        print("   Page 5: Cost per Image")
        print("   Page 6: Cumulative Cost Over Time")
        print("   Page 7: Cost Savings per Image")
        print("   Page 8: Image Quality Winners (CLIP)")
        print("   Page 9: Detailed CLIP Scores")
        print("   Page 10: Quality vs. Cost Analysis")
        print("   Page 11: Prompt Quality Evaluation")
        print("="*80)
        print("💡 Added new plots for cumulative cost and quality-vs-cost analysis!")
        print("="*80)


