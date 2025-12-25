"""
Experiment comparison tool for analyzing multiple model runs.
Creates comparative visualizations and summary reports.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ExperimentComparator:
    """
    Compare results from multiple experiments and create summary reports.
    """
    
    def __init__(self, results_dir: str = './results'):
        """
        Initialize experiment comparator.
        
        Args:
            results_dir: Base results directory
        """
        self.results_dir = results_dir
    
    def load_experiment_metrics(self, experiment_folder: str) -> Dict:
        """
        Load metrics from an experiment folder.
        
        Args:
            experiment_folder: Path to experiment folder
            
        Returns:
            Dictionary with experiment info and metrics
        """
        metrics_file = os.path.join(experiment_folder, 'data', 'metrics.json')
        
        if not os.path.exists(metrics_file):
            return None
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        exp_name = os.path.basename(experiment_folder)
        return {
            'name': exp_name,
            'metrics': data.get('metrics', {}),
            'timestamp': data.get('timestamp', 'Unknown'),
            'path': experiment_folder
        }
    
    def collect_all_experiments(self) -> List[Dict]:
        """
        Collect metrics from all experiments in results directory.
        
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        
        if not os.path.exists(self.results_dir):
            print(f"Results directory not found: {self.results_dir}")
            return experiments
        
        for exp_folder in os.listdir(self.results_dir):
            exp_path = os.path.join(self.results_dir, exp_folder)
            
            if os.path.isdir(exp_path):
                exp_data = self.load_experiment_metrics(exp_path)
                if exp_data:
                    experiments.append(exp_data)
        
        return experiments
    
    def create_comparison_table(self, experiments: List[Dict]) -> pd.DataFrame:
        """
        Create a comparison table with all experiments and metrics.
        
        Args:
            experiments: List of experiment dictionaries
            
        Returns:
            DataFrame with comparison
        """
        if not experiments:
            return pd.DataFrame()
        
        # Extract all unique metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp['metrics'].keys())
        
        all_metrics = sorted(all_metrics)
        
        # Build dataframe
        data = []
        for exp in experiments:
            row = {'Experiment': exp['name']}
            for metric in all_metrics:
                row[metric] = exp['metrics'].get(metric, np.nan)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def create_comparison_plot(self, 
                              experiments: List[Dict],
                              metrics: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Create interactive comparison plot for multiple experiments.
        
        Args:
            experiments: List of experiment dictionaries
            metrics: List of metrics to compare (None = all)
            save_path: Path to save plot
        """
        if not experiments:
            print("No experiments to compare")
            return
        
        # Prepare data
        df = self.create_comparison_table(experiments)
        
        if metrics is None:
            # Use common metrics
            metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
            metrics = [m for m in metrics if m in df.columns]
        
        # Filter to requested metrics
        plot_metrics = [m for m in metrics if m in df.columns]
        
        if not plot_metrics:
            print("No valid metrics found for comparison")
            return
        
        # Create subplots - one for each metric
        n_metrics = len(plot_metrics)
        rows = (n_metrics + 1) // 2
        cols = 2 if n_metrics > 1 else 1
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=plot_metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, metric in enumerate(plot_metrics):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            # Get values and experiment names
            values = df[metric].values
            names = df['Experiment'].values
            
            # Create bar chart
            fig.add_trace(
                go.Bar(
                    x=names,
                    y=values,
                    marker_color=colors[idx % len(colors)],
                    text=[f'{v:.4f}' if not np.isnan(v) else 'N/A' for v in values],
                    textposition='auto',
                    showlegend=False,
                    name=metric
                ),
                row=row, col=col
            )
            
            # Update axes
            fig.update_xaxes(tickangle=-45, row=row, col=col)
            fig.update_yaxes(title_text=metric, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=dict(text='Experiment Comparison', font=dict(size=20)),
            template='plotly_white',
            height=400 * rows,
            showlegend=False
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"📊 Comparison plot saved to: {save_path}")
        else:
            fig.show()
    
    def create_ranking_plot(self,
                           experiments: List[Dict],
                           metric: str = 'MAE',
                           ascending: bool = True,
                           save_path: Optional[str] = None) -> None:
        """
        Create ranking visualization for a specific metric.
        
        Args:
            experiments: List of experiment dictionaries
            metric: Metric to rank by
            ascending: Whether lower is better
            save_path: Path to save plot
        """
        if not experiments:
            print("No experiments to rank")
            return
        
        # Extract metric values
        data = []
        for exp in experiments:
            value = exp['metrics'].get(metric, np.nan)
            if not np.isnan(value):
                data.append({
                    'Experiment': exp['name'],
                    'Value': value
                })
        
        if not data:
            print(f"No valid data found for metric: {metric}")
            return
        
        # Sort
        df = pd.DataFrame(data).sort_values('Value', ascending=ascending)
        df['Rank'] = range(1, len(df) + 1)
        
        # Create plot
        fig = go.Figure()
        
        # Color gradient based on rank
        colors = px.colors.sequential.Viridis
        color_scale = [colors[int(i * (len(colors) - 1) / max(1, len(df) - 1))] 
                      for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df['Experiment'],
            y=df['Value'],
            marker_color=color_scale,
            text=[f"#{r}: {v:.6f}" for r, v in zip(df['Rank'], df['Value'])],
            textposition='auto',
        ))
        
        # Add ranking labels
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['Experiment'],
                y=row['Value'],
                text=f"#{row['Rank']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                ax=0,
                ay=-40
            )
        
        fig.update_layout(
            title=dict(
                text=f'Experiment Ranking by {metric} {"(Lower is Better)" if ascending else "(Higher is Better)"}',
                font=dict(size=18)
            ),
            xaxis_title="Experiment",
            yaxis_title=metric,
            template='plotly_white',
            height=600,
            xaxis_tickangle=-45
        )
        
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
            print(f"📊 Ranking plot saved to: {save_path}")
        else:
            fig.show()
    
    def create_markdown_summary(self,
                               experiments: List[Dict],
                               save_path: Optional[str] = None) -> str:
        """
        Create markdown summary of all experiments.
        
        Args:
            experiments: List of experiment dictionaries
            save_path: Path to save markdown file
            
        Returns:
            Markdown content
        """
        if not experiments:
            return "# No experiments found"
        
        df = self.create_comparison_table(experiments)
        
        # Create markdown
        md = f"""# Experiment Comparison Summary

**Total Experiments:** {len(experiments)}  
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Overall Results

{df.to_markdown(index=False, floatfmt='.6f')}

---

## Best Models by Metric

"""
        
        # Find best for each metric
        for col in df.columns:
            if col != 'Experiment':
                # Assume lower is better for most metrics except CORR
                ascending = 'CORR' not in col
                best_idx = df[col].idxmin() if ascending else df[col].idxmax()
                best_exp = df.loc[best_idx, 'Experiment']
                best_val = df.loc[best_idx, col]
                
                if not pd.isna(best_val):
                    md += f"- **{col}**: {best_exp} ({best_val:.6f})\n"
        
        md += "\n---\n\n## Experiment Details\n\n"
        
        # Add details for each experiment
        for exp in sorted(experiments, key=lambda x: x['name']):
            md += f"### {exp['name']}\n\n"
            md += f"- **Path**: `{exp['path']}`\n"
            md += f"- **Timestamp**: {exp['timestamp']}\n"
            md += "\n**Metrics:**\n\n"
            
            metrics_data = []
            for metric_name, value in sorted(exp['metrics'].items()):
                if isinstance(value, float):
                    metrics_data.append({'Metric': metric_name, 'Value': f'{value:.6f}'})
                else:
                    metrics_data.append({'Metric': metric_name, 'Value': str(value)})
            
            metrics_df = pd.DataFrame(metrics_data)
            md += metrics_df.to_markdown(index=False)
            md += "\n\n---\n\n"
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(md)
            print(f"📄 Summary saved to: {save_path}")
        
        return md
    
    def generate_full_comparison_report(self,
                                       output_dir: Optional[str] = None) -> str:
        """
        Generate a full comparison report with all visualizations.
        
        Args:
            output_dir: Directory to save report (default: results/comparison)
            
        Returns:
            Path to output directory
        """
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, '_comparison_report')
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("📊 Generating Experiment Comparison Report".center(80))
        print(f"{'='*80}\n")
        
        # Collect all experiments
        print("🔍 Collecting experiments...")
        experiments = self.collect_all_experiments()
        
        if not experiments:
            print("❌ No experiments found in results directory")
            return output_dir
        
        print(f"✓ Found {len(experiments)} experiments\n")
        
        # Create comparison table
        print("📋 Creating comparison table...")
        df = self.create_comparison_table(experiments)
        df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
        print(f"✓ Saved to: {os.path.join(output_dir, 'comparison_table.csv')}\n")
        
        # Create visualizations
        print("📊 Creating visualizations...")
        
        self.create_comparison_plot(
            experiments,
            save_path=os.path.join(output_dir, 'metric_comparison.html')
        )
        
        # Create ranking for key metrics
        for metric in ['MAE', 'RMSE', 'MAPE']:
            if metric in df.columns and not df[metric].isna().all():
                self.create_ranking_plot(
                    experiments,
                    metric=metric,
                    save_path=os.path.join(output_dir, f'ranking_{metric.lower()}.html')
                )
        
        # Create markdown summary
        print("\n📝 Creating markdown summary...")
        self.create_markdown_summary(
            experiments,
            save_path=os.path.join(output_dir, 'SUMMARY.md')
        )
        
        print(f"\n{'='*80}")
        print("✨ Comparison Report Complete!".center(80))
        print(f"{'='*80}")
        print(f"\n📂 Location: {output_dir}")
        print(f"📄 Summary: {os.path.join(output_dir, 'SUMMARY.md')}")
        print(f"📊 Visualizations: {output_dir}/*.html\n")
        
        return output_dir


def main():
    """Command-line interface for experiment comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare time series forecasting experiments')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Results directory to analyze')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison report')
    
    args = parser.parse_args()
    
    comparator = ExperimentComparator(results_dir=args.results_dir)
    comparator.generate_full_comparison_report(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
