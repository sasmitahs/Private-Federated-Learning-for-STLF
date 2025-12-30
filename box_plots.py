import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

class MetricsBoxPlotGenerator:
    def __init__(self, metrics_dir, predictions_dir, output_dir, model_name="dual_cnn_gru_fcnn"):
        """
        Initialize the box plot generator.
        
        Args:
            metrics_dir: Directory containing metric CSV files
            predictions_dir: Directory containing prediction CSV files
            output_dir: Directory to save generated plots
            model_name: Name of the model to analyze
        """
        self.metrics_dir = metrics_dir
        self.predictions_dir = predictions_dir
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)
    
    def load_metrics_for_strategy(self, strategy):
        """Load all metric files for a given strategy."""
        pattern = f"cid*_{self.model_name}_{strategy}_metrics.csv"
        metric_files = list(Path(self.metrics_dir).glob(pattern))
        
        all_metrics = []
        errors = []
        
        for file in metric_files:
            try:
                df = pd.read_csv(file)
                
                # Extract CID from filename
                cid = int(file.stem.split('_')[0].replace('cid', ''))
                
                # If NRMSE is missing, calculate it from predictions
                if 'NRMSE' not in df.columns and 'RMSE' in df.columns:
                    pred_file = os.path.join(self.predictions_dir, f"{cid}_{self.model_name}_{strategy}.csv")
                    if os.path.exists(pred_file):
                        pred_df = pd.read_csv(pred_file)
                        pred_df = pred_df.dropna(subset=['true', 'pred'])
                        y_true = pred_df['true'].values
                        if len(y_true) > 0:
                            true_range = np.max(y_true) - np.min(y_true)
                            rmse = df['RMSE'].iloc[0]
                            nrmse = rmse / true_range if true_range != 0 else np.nan
                            df['NRMSE'] = nrmse
                        else:
                            df['NRMSE'] = np.nan
                    else:
                        df['NRMSE'] = np.nan
                        errors.append(f"{file.name}: Prediction file missing for NRMSE calculation - {pred_file}")
                
                # Check if we have valid data for required metrics
                required_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']
                if not df.empty and not df[required_metrics].isna().all().all():
                    df['CID'] = cid
                    df['Strategy'] = strategy
                    all_metrics.append(df)
                else:
                    errors.append(f"{file.name}: Empty or all NaN values")
                    
            except KeyError as e:
                errors.append(f"{file.name}: Missing columns - {e}")
            except Exception as e:
                errors.append(f"{file.name}: {e}")
        
        if errors and len(errors) <= 10:
            print(f"  Errors encountered:")
            for err in errors:
                print(f"    - {err}")
        elif errors:
            print(f"  {len(errors)} files had errors (first 5 shown):")
            for err in errors[:5]:
                print(f"    - {err}")
                
        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        return pd.DataFrame()
    
    def load_all_strategies(self, strategies):
        """Load metrics for all strategies."""
        all_data = []
        for strategy in strategies:
            print(f"\nLoading metrics for strategy: {strategy}")
            df = self.load_metrics_for_strategy(strategy)
            if not df.empty:
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} records")
            else:
                print(f"  ✗ No data found")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def create_boxplot(self, data, metric, title, ylabel, filename, remove_outliers=False):
        """Create a single box plot for a metric across strategies."""
        # Filter out NaN values for this metric
        plot_data = data[data[metric].notna()].copy()
        
        if plot_data.empty:
            print(f"  Warning: No valid data for {metric}, skipping plot")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Remove outliers if specified (for MAPE which can have extreme values)
        if remove_outliers:
            Q1 = plot_data.groupby('Strategy')[metric].transform('quantile', 0.25)
            Q3 = plot_data.groupby('Strategy')[metric].transform('quantile', 0.75)
            IQR = Q3 - Q1
            plot_data = plot_data[~((plot_data[metric] < (Q1 - 1.5 * IQR)) | (plot_data[metric] > (Q3 + 1.5 * IQR)))]
        
        # Create box plot
        ax = sns.boxplot(data=plot_data, x='Strategy', y=metric, palette='Set2', showfliers=True)
        
        # Add median values on top of boxes
        medians = plot_data.groupby('Strategy')[metric].median()
        positions = range(len(medians))
        for pos, (strategy, median) in enumerate(medians.items()):
            ax.text(pos, median, f'{median:.2f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Styling
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Strategy', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    def create_combined_boxplot(self, data, metrics_config, filename):
        """Create a combined subplot with multiple metrics."""
        n_metrics = len(metrics_config)
        
        # Adjust subplot layout based on number of metrics
        if n_metrics <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for metric, config in metrics_config.items():
            # Filter out NaN values for this metric
            plot_data = data[data[metric].notna()].copy()
            
            if plot_data.empty:
                print(f"  Warning: No valid data for {metric} in combined plot, skipping")
                continue
            
            ax = axes[plot_idx]
            
            # Handle outliers for MAPE
            if config.get('remove_outliers', False):
                Q1 = plot_data.groupby('Strategy')[metric].transform('quantile', 0.25)
                Q3 = plot_data.groupby('Strategy')[metric].transform('quantile', 0.75)
                IQR = Q3 - Q1
                plot_data = plot_data[~((plot_data[metric] < (Q1 - 1.5 * IQR)) | 
                                       (plot_data[metric] > (Q3 + 1.5 * IQR)))]
            
            # Create box plot
            sns.boxplot(data=plot_data, x='Strategy', y=metric, 
                       palette='Set2', ax=ax, showfliers=True)
            
            # Add median values
            medians = plot_data.groupby('Strategy')[metric].median()
            positions = range(len(medians))
            for pos, (strategy, median) in enumerate(medians.items()):
                ax.text(pos, median, f'{median:.2f}', 
                       ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            # Styling
            ax.set_title(config['title'], fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Strategy', fontsize=10, fontweight='bold')
            ax.set_ylabel(config['ylabel'], fontsize=10, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            
            plot_idx += 1
        
        # Remove extra subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'Metrics Comparison - {self.model_name}', 
                    fontsize=16, fontweight='bold', y=0.998)
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    def create_strategy_comparison_table(self, data, strategies, output_filename):
        """Create a comparison table with median metrics for all strategies."""
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']
        
        comparison_data = []
        for strategy in strategies:
            strategy_data = data[data['Strategy'] == strategy]
            if not strategy_data.empty:
                row = {'Strategy': strategy}
                for metric in metrics:
                    if metric in strategy_data.columns:
                        valid_data = strategy_data[metric].dropna()
                        if len(valid_data) > 0:
                            row[f'{metric}_median'] = valid_data.median()
                            row[f'{metric}_std'] = valid_data.std()
                            row[f'{metric}_count'] = len(valid_data)
                        else:
                            row[f'{metric}_median'] = np.nan
                            row[f'{metric}_std'] = np.nan
                            row[f'{metric}_count'] = 0
                    else:
                        row[f'{metric}_median'] = np.nan
                        row[f'{metric}_std'] = np.nan
                        row[f'{metric}_count'] = 0
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, output_filename)
        comparison_df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved comparison table: {csv_path}")
        
        return comparison_df
    
    def generate_all_plots(self, strategies):
        """Generate all box plots and comparison tables."""
        print(f"\n{'='*70}")
        print(f"  GENERATING BOX PLOTS FOR MODEL: {self.model_name}")
        print(f"{'='*70}\n")
        
        # Load all data
        print("Loading metrics data...")
        data = self.load_all_strategies(strategies)
        
        if data.empty:
            print("\n✗ No data found. Exiting.")
            return
        
        print(f"\n{'='*70}")
        print(f"DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Total records loaded: {len(data)}")
        print(f"Strategies found: {', '.join(data['Strategy'].unique())}")
        print(f"\nRecords per strategy:")
        for strategy, count in data.groupby('Strategy').size().items():
            print(f"  {strategy}: {count}")
        
        # Check data availability per metric
        print(f"\nData availability per metric:")
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']
        for metric in metrics:
            valid_count = data[metric].notna().sum()
            print(f"  {metric}: {valid_count}/{len(data)} ({100*valid_count/len(data):.1f}%)")
        
        # Define metrics configuration
        metrics_config = {
            'MAE': {
                'title': 'Mean Absolute Error (MAE)',
                'ylabel': 'MAE',
                'remove_outliers': False
            },
            'MSE': {
                'title': 'Mean Squared Error (MSE)',
                'ylabel': 'MSE',
                'remove_outliers': False
            },
            'RMSE': {
                'title': 'Root Mean Squared Error (RMSE)',
                'ylabel': 'RMSE',
                'remove_outliers': False
            },
            'MAPE (%)': {
                'title': 'Mean Absolute Percentage Error (MAPE)',
                'ylabel': 'MAPE (%)',
                'remove_outliers': True  # MAPE often has extreme outliers
            },
            'SMAPE (%)': {
                'title': 'Symmetric Mean Absolute Percentage Error (SMAPE)',
                'ylabel': 'SMAPE (%)',
                'remove_outliers': False
            },
            'NRMSE': {
                'title': 'Normalized Root Mean Squared Error (NRMSE)',
                'ylabel': 'NRMSE',
                'remove_outliers': False
            }
        }
        
        # Create individual plots
        print(f"\n{'='*70}")
        print("GENERATING INDIVIDUAL METRIC PLOTS")
        print(f"{'='*70}\n")
        for metric, config in metrics_config.items():
            print(f"Creating plot for {metric}...")
            self.create_boxplot(
                data=data,
                metric=metric,
                title=config['title'],
                ylabel=config['ylabel'],
                filename=f"{self.model_name}_{metric.replace(' ', '_').replace('(%)', '').replace('(', '').replace(')', '')}_boxplot.png",
                remove_outliers=config.get('remove_outliers', False)
            )
        
        # Create combined plot
        print(f"\n{'='*70}")
        print("GENERATING COMBINED METRICS PLOT")
        print(f"{'='*70}\n")
        self.create_combined_boxplot(
            data=data,
            metrics_config=metrics_config,
            filename=f"{self.model_name}_all_metrics_combined.png"
        )
        
        # Create comparison table
        print(f"\n{'='*70}")
        print("GENERATING COMPARISON TABLE")
        print(f"{'='*70}\n")
        comparison_df = self.create_strategy_comparison_table(
            data=data,
            strategies=strategies,
            output_filename=f"{self.model_name}_metrics_comparison.csv"
        )
        
        # Print summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS - MEDIAN VALUES BY STRATEGY")
        print(f"{'='*70}\n")
        
        # Format output
        summary_cols = ['Strategy', 'MAE_median', 'RMSE_median', 'SMAPE (%)_median', 'NRMSE_median']
        available_cols = [col for col in summary_cols if col in comparison_df.columns]
        
        if available_cols:
            summary_df = comparison_df[available_cols].copy()
            
            # Rename columns for display
            display_names = {
                'Strategy': 'Strategy',
                'MAE_median': 'MAE',
                'RMSE_median': 'RMSE',
                'SMAPE (%)_median': 'SMAPE(%)',
                'NRMSE_median': 'NRMSE'
            }
            summary_df = summary_df.rename(columns={k: v for k, v in display_names.items() if k in summary_df.columns})
            
            print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Also show best performing strategy for each metric
        print(f"\n{'='*70}")
        print("BEST PERFORMING STRATEGIES (LOWEST MEDIAN)")
        print(f"{'='*70}\n")
        for metric in ['MAE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'NRMSE']:
            col_name = f'{metric}_median'
            if col_name in comparison_df.columns:
                best_idx = comparison_df[col_name].idxmin()
                best_strategy = comparison_df.loc[best_idx, 'Strategy']
                best_value = comparison_df.loc[best_idx, col_name]
                print(f"  {metric:12s}: {best_strategy:35s} ({best_value:.4f})")
        
        print(f"\n{'='*70}")
        print("✓ BOX PLOTS GENERATION COMPLETED!")
        print(f"✓ All outputs saved to: {self.output_dir}")
        print(f"{'='*70}\n")





# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Configuration
    METRICS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"
    PREDICTIONS_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/predictions40-50-168-T"
    OUTPUT_DIR = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/boxplots"
    MODEL_NAME = "dual_cnn_gru_fcnn"
    
    # Define all strategies from ISP-1 results
    STRATEGIES = [
        "AEpublic_k-means_2enc",
        "no-cluster_no-AE",
        "no-cluster_no-AE_FedProx",
        "AEpublic_k-means_SCAFFOLD",
        "AEpublic_k-means_FedProx",
        "poc",
        
    ]
    
    # Create generator instance
    generator = MetricsBoxPlotGenerator(
        metrics_dir=METRICS_DIR,
        predictions_dir=PREDICTIONS_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME
    )
    
    # Generate all plots
    generator.generate_all_plots(STRATEGIES)