import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List

# Set up logging
log_dir = '/home/user/DPFL-Sasmita/FL-Baseline-Codes/logs/'
os.makedirs(log_dir, exist_ok=True)
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(log_dir, f"convergence_log_{timestamp}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_metrics_data(
    median_client: int,
    model_name: str,
    strategies: Dict[str, str],
    metrics_dir: str
) -> Dict[str, pd.DataFrame]:
    """
    Load metrics data from existing CSV files for the specified client and strategies.
    Returns a dict of strategy -> metrics DataFrame.
    """
    metrics_data = {}
    
    for strat_name, aggr_strat in strategies.items():
        metrics_csv = os.path.join(metrics_dir, f"cid{median_client}_{model_name}_{aggr_strat}_metrics_all_rounds.csv")
        if os.path.exists(metrics_csv):
            try:
                df = pd.read_csv(metrics_csv)
                if not df.empty:
                    metrics_data[strat_name] = df
                    logging.info(f"Loaded metrics for {strat_name} from {metrics_csv}")
                else:
                    logging.warning(f"Empty metrics file for {strat_name}: {metrics_csv}")
            except Exception as e:
                logging.error(f"Failed to load {metrics_csv}: {e}")
        else:
            logging.warning(f"Metrics file not found for {strat_name}: {metrics_csv}")
    
    return metrics_data

def plot_convergence_graphs(
    median_client: int,
    model_name: str,
    strategies: Dict[str, str],
    rounds: List[int],
    metrics_data: Dict[str, pd.DataFrame],
    save_dir: str = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/convergence_plots"
):
    """
    Plot convergence graphs for NRMSE and SMAPE for the median client across all strategies,
    with a box around the legend for strategies containing 'clustering' and increased font sizes.
    """
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Plotting convergence graphs for CID={median_client} using model {model_name}")
    
    # Define metrics to plot (only NRMSE and SMAPE)
    metrics_to_plot = ["NRMSE", "SMAPE (%)"]
    
    # Validate metrics data
    valid_strategies = []
    for strat_name, df in metrics_data.items():
        if df.empty or all(df[metric].isna().all() for metric in metrics_to_plot):
            logging.warning(f"No valid data for strategy {strat_name}. Skipping.")
        else:
            valid_strategies.append(strat_name)
    
    if not valid_strategies:
        logging.error("No valid data available for any strategy. Cannot generate plots.")
        return
    
    # Plot each metric separately
    for metric in metrics_to_plot:
        plt.figure(figsize=(16, 12))  # Increased figure size
        for strat_name in valid_strategies:
            df = metrics_data[strat_name]
            if 'round' in df.columns and metric in df.columns:
                df = df.sort_values('round')  # Ensure rounds are sorted
                metric_values = df[metric].values
                plot_rounds = df['round'].values
                if len(metric_values) == len(plot_rounds):
                    plt.plot(plot_rounds, metric_values, label=strat_name, marker='o', markersize=6, linewidth=2.5)
                else:
                    logging.warning(f"Incomplete {metric} data for {strat_name}. Expected {len(rounds)} rounds, got {len(metric_values)}.")
            else:
                logging.warning(f"Missing 'round' or '{metric}' column in data for {strat_name}.")
        
        # Customize plot
        plt.xlabel('Round', fontsize=40, weight='bold')
        plt.ylabel(metric, fontsize=40, weight='bold')
        plt.title(f'Convergence of {metric} for Client {median_client} (DualEncDecoder)', fontsize=40, weight='bold')
        plt.grid(True)
        
        # Customize legend with box for strategies containing 'clustering'
        legend = plt.legend(fontsize=40, loc='best')
        if any('clustering' in strat_name.lower() for strat_name in valid_strategies):
            legend.get_frame().set_linewidth(2.0)  # Add box around legend
            legend.get_frame().set_edgecolor('black')
        
        # Add border to the plot
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_color('black')
        
        # Increase tick label sizes
        plt.xticks(fontsize=40, weight='bold')
        plt.yticks(fontsize=40, weight='bold')
        plt.tight_layout()
        
        # Save plot
        metric_filename = metric.lower().replace(' (%)', '_pct').replace(' ', '_')
        plot_path = os.path.join(save_dir, f"convergence_{metric_filename}_client_{median_client}_dualencdecoder.png")
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved {metric} convergence plot to {plot_path}")
        except Exception as e:
            logging.error(f"Failed to save {metric} plot: {e}")
        finally:
            plt.close()

if __name__ == "__main__":
    # Configuration
    model_name = "dual_cnn_gru_fcnn"
    metrics_dir = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/metrics40-50-168-T"
    rounds = list(range(1, 41))  # Rounds 1 to 40
    save_dir = "/home/user/DPFL-Sasmita/FL-Baseline-Codes/convergence_plots"
    
    strategies = {
        "POC_with_clustering": "poc_correct",
        "POC_without_clustering": "poc_nocluster",
        "DAS_with_clustering": "AEpublic_k-means_2enc_correct",
        "DAS_without_clustering": "no-cluster_no-AE_correct"
    }
    
    # Specify the median client ID (e.g., 101 based on your file)
    median_client = 101  # Adjust this if you want a different client ID
    
    # Step 1: Load existing metrics data
    metrics_data = load_metrics_data(
        median_client=median_client,
        model_name=model_name,
        strategies=strategies,
        metrics_dir=metrics_dir
    )
    
    # Step 2: Plot convergence graphs for NRMSE and SMAPE
    plot_convergence_graphs(
        median_client=median_client,
        model_name=model_name,
        strategies=strategies,
        rounds=rounds,
        metrics_data=metrics_data,
        save_dir=save_dir
    )
    
    print(f"Convergence plots saved to {save_dir}/convergence_nrmse_client_{median_client}_dualencdecoder.png")
    print(f"Convergence plots saved to {save_dir}/convergence_smape_pct_client_{median_client}_dualencdecoder.png")