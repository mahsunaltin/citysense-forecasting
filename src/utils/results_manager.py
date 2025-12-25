"""
Comprehensive results manager for time series forecasting experiments.
Creates organized outputs with visualizations, metrics, and reports.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, List, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ResultsManager:
    """
    Manages experiment results with comprehensive outputs:
    - Interactive Plotly visualizations
    - Markdown reports with metrics tables
    - JSON metadata
    - Organized file structure
    """
    
    def __init__(self, base_results_dir: str = './results'):
        """
        Initialize results manager.
        
        Args:
            base_results_dir: Base directory for all results
        """
        self.base_results_dir = base_results_dir
        os.makedirs(base_results_dir, exist_ok=True)
        
    def create_experiment_folder(self, setting: str) -> str:
        """
        Create organized folder structure for experiment.
        
        Structure:
        results/
            {setting}/
                plots/
                data/
                reports/
                metadata.json
                
        Args:
            setting: Experiment setting string
            
        Returns:
            Path to experiment folder
        """
        exp_folder = os.path.join(self.base_results_dir, setting)
        
        # Create subfolders
        plots_dir = os.path.join(exp_folder, 'plots')
        data_dir = os.path.join(exp_folder, 'data')
        reports_dir = os.path.join(exp_folder, 'reports')
        
        for folder in [exp_folder, plots_dir, data_dir, reports_dir]:
            os.makedirs(folder, exist_ok=True)
        
        return exp_folder
    
    def save_predictions(self, 
                        preds: np.ndarray, 
                        trues: np.ndarray, 
                        inputs: Optional[np.ndarray],
                        exp_folder: str) -> None:
        """
        Save prediction arrays in organized manner.
        
        Args:
            preds: Predictions array
            trues: Ground truth array
            inputs: Input sequences array (optional)
            exp_folder: Experiment folder path
        """
        data_dir = os.path.join(exp_folder, 'data')
        
        np.save(os.path.join(data_dir, 'predictions.npy'), preds)
        np.save(os.path.join(data_dir, 'ground_truth.npy'), trues)
        
        if inputs is not None:
            np.save(os.path.join(data_dir, 'inputs.npy'), inputs)
        
        # Also save as CSV for easy inspection
        # Flatten for CSV (first sample, all features)
        pd.DataFrame(preds[0]).to_csv(
            os.path.join(data_dir, 'predictions_sample.csv'), 
            index=False
        )
        pd.DataFrame(trues[0]).to_csv(
            os.path.join(data_dir, 'ground_truth_sample.csv'), 
            index=False
        )
    
    def save_metrics(self, 
                     metrics: Dict[str, float], 
                     exp_folder: str,
                     additional_info: Optional[Dict] = None) -> None:
        """
        Save metrics as JSON and numpy array.
        
        Args:
            metrics: Dictionary of metrics
            exp_folder: Experiment folder path
            additional_info: Additional metadata to save
        """
        data_dir = os.path.join(exp_folder, 'data')
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save as JSON
        metrics_with_info = {
            'timestamp': datetime.now().isoformat(),
            'metrics': convert_to_json_serializable(metrics)
        }
        
        if additional_info:
            metrics_with_info.update(convert_to_json_serializable(additional_info))
        
        with open(os.path.join(data_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics_with_info, f, indent=4)
        
        # Save core metrics as numpy for compatibility
        core_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
        metric_values = [metrics.get(m, 0.0) for m in core_metrics]
        np.save(os.path.join(data_dir, 'metrics.npy'), np.array(metric_values))
    
    def create_prediction_plot(self,
                              preds: np.ndarray,
                              trues: np.ndarray,
                              inputs: Optional[np.ndarray],
                              exp_folder: str,
                              sample_idx: int = 0,
                              feature_idx: int = -1,
                              title: Optional[str] = None) -> str:
        """
        Create interactive prediction plot using Plotly.
        
        Args:
            preds: Predictions array
            trues: Ground truth array
            inputs: Input sequences (optional)
            exp_folder: Experiment folder path
            sample_idx: Sample index to plot
            feature_idx: Feature index to plot
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        plots_dir = os.path.join(exp_folder, 'plots')
        
        # Extract data for specific sample and feature
        pred_seq = preds[sample_idx, :, feature_idx]
        true_seq = trues[sample_idx, :, feature_idx]
        
        fig = go.Figure()
        
        # If inputs provided, plot them first
        if inputs is not None:
            input_seq = inputs[sample_idx, :, feature_idx] if inputs.ndim > 2 else inputs[sample_idx, :]
            input_len = len(input_seq)
            
            fig.add_trace(go.Scatter(
                x=list(range(input_len)),
                y=input_seq,
                mode='lines',
                name='Input Sequence',
                line=dict(color='#2E86AB', width=2.5),
                opacity=0.8
            ))
            
            # Plot predictions starting after input
            pred_range = list(range(input_len, input_len + len(pred_seq)))
            
            fig.add_vline(
                x=input_len,
                line_dash="dot",
                line_color="gray",
                opacity=0.6,
                annotation_text="Forecast Horizon",
                annotation_position="top"
            )
        else:
            pred_range = list(range(len(pred_seq)))
        
        # Ground truth
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=true_seq,
            mode='lines',
            name='Ground Truth',
            line=dict(color='#06A77D', width=3)
        ))
        
        # Predictions
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=pred_seq,
            mode='lines',
            name='Prediction',
            line=dict(color='#D62246', width=2.5, dash='dash')
        ))
        
        # Calculate sample metrics
        mae = np.mean(np.abs(pred_seq - true_seq))
        rmse = np.sqrt(np.mean((pred_seq - true_seq) ** 2))
        
        plot_title = title or f'Prediction vs Ground Truth (Sample {sample_idx}, Feature {feature_idx})'
        
        fig.update_layout(
            title=dict(
                text=f'{plot_title}<br><sub>MAE: {mae:.4f} | RMSE: {rmse:.4f}</sub>',
                font=dict(size=16)
            ),
            xaxis_title="Time Step",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        save_path = os.path.join(plots_dir, f'prediction_sample_{sample_idx}_feature_{feature_idx}.html')
        fig.write_html(save_path)
        
        return save_path
    
    def create_error_analysis_plot(self,
                                   preds: np.ndarray,
                                   trues: np.ndarray,
                                   exp_folder: str) -> str:
        """
        Create comprehensive error analysis plots.
        
        Args:
            preds: Predictions array
            trues: Ground truth array
            exp_folder: Experiment folder path
            
        Returns:
            Path to saved plot
        """
        plots_dir = os.path.join(exp_folder, 'plots')
        
        errors = preds - trues
        errors_flat = errors.flatten()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Error Distribution',
                'Error Over Time (Mean ± Std)',
                'Absolute Error by Horizon',
                'Prediction vs True Scatter'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.12
        )
        
        # 1. Error histogram
        fig.add_trace(
            go.Histogram(
                x=errors_flat,
                nbinsx=50,
                marker_color='#2E86AB',
                opacity=0.7,
                name='Error Distribution',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Error over time (mean across samples)
        mean_errors = errors.mean(axis=0).mean(axis=-1)  # Mean across samples and features
        std_errors = errors.std(axis=0).mean(axis=-1)
        time_steps = list(range(len(mean_errors)))
        
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=mean_errors,
                mode='lines',
                name='Mean Error',
                line=dict(color='#D62246', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=time_steps + time_steps[::-1],
                y=np.concatenate([mean_errors + std_errors, (mean_errors - std_errors)[::-1]]),
                fill='toself',
                fillcolor='rgba(214, 34, 70, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='±1 Std'
            ),
            row=1, col=2
        )
        
        # 3. Absolute error by forecast horizon
        abs_errors = np.abs(errors).mean(axis=0).mean(axis=-1)
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=abs_errors,
                mode='lines+markers',
                name='MAE by Horizon',
                line=dict(color='#06A77D', width=2.5),
                marker=dict(size=5),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 4. Prediction vs True scatter
        # Sample for visualization (too many points can be slow)
        sample_size = min(10000, len(preds.flatten()))
        indices = np.random.choice(len(preds.flatten()), sample_size, replace=False)
        
        preds_sample = preds.flatten()[indices]
        trues_sample = trues.flatten()[indices]
        
        fig.add_trace(
            go.Scatter(
                x=trues_sample,
                y=preds_sample,
                mode='markers',
                marker=dict(
                    size=3,
                    color='#2E86AB',
                    opacity=0.5
                ),
                name='Predictions',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Perfect prediction line
        min_val = min(trues_sample.min(), preds_sample.min())
        max_val = max(trues_sample.max(), preds_sample.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Prediction',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Error", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Time Step", row=1, col=2)
        fig.update_yaxes(title_text="Error", row=1, col=2)
        
        fig.update_xaxes(title_text="Forecast Horizon", row=2, col=1)
        fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=1)
        
        fig.update_xaxes(title_text="Ground Truth", row=2, col=2)
        fig.update_yaxes(title_text="Prediction", row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title=dict(text='Error Analysis', font=dict(size=18)),
            template='plotly_white',
            height=800,
            showlegend=False
        )
        
        save_path = os.path.join(plots_dir, 'error_analysis.html')
        fig.write_html(save_path)
        
        return save_path
    
    def create_feature_comparison_plot(self,
                                      preds: np.ndarray,
                                      trues: np.ndarray,
                                      exp_folder: str,
                                      feature_names: Optional[List[str]] = None,
                                      max_features: int = 6) -> str:
        """
        Create multi-feature comparison plot.
        
        Args:
            preds: Predictions array (samples, timesteps, features)
            trues: Ground truth array
            exp_folder: Experiment folder path
            feature_names: Names of features
            max_features: Maximum features to plot
            
        Returns:
            Path to saved plot
        """
        plots_dir = os.path.join(exp_folder, 'plots')
        
        n_features = min(preds.shape[-1], max_features)
        
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(n_features)]
        
        # Create subplots
        fig = make_subplots(
            rows=n_features,
            cols=1,
            subplot_titles=feature_names[:n_features],
            vertical_spacing=0.05
        )
        
        # Use first sample for visualization
        sample_idx = 0
        
        for idx in range(n_features):
            pred_seq = preds[sample_idx, :, idx]
            true_seq = trues[sample_idx, :, idx]
            
            # Ground truth
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(true_seq))),
                    y=true_seq,
                    mode='lines',
                    name='Ground Truth' if idx == 0 else None,
                    showlegend=(idx == 0),
                    line=dict(color='#06A77D', width=2),
                    legendgroup='gt'
                ),
                row=idx+1, col=1
            )
            
            # Prediction
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(pred_seq))),
                    y=pred_seq,
                    mode='lines',
                    name='Prediction' if idx == 0 else None,
                    showlegend=(idx == 0),
                    line=dict(color='#D62246', width=2, dash='dash'),
                    legendgroup='pred'
                ),
                row=idx+1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=dict(text='Multi-Feature Predictions', font=dict(size=18)),
            template='plotly_white',
            height=300 * n_features,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Update axes
        for i in range(1, n_features + 1):
            fig.update_xaxes(title_text="Time Step", row=i, col=1)
            fig.update_yaxes(title_text="Value", row=i, col=1)
        
        save_path = os.path.join(plots_dir, 'feature_comparison.html')
        fig.write_html(save_path)
        
        return save_path
    
    def create_metrics_summary_plot(self,
                                   metrics: Dict[str, float],
                                   exp_folder: str) -> str:
        """
        Create visual summary of all metrics.
        
        Args:
            metrics: Dictionary of metrics
            exp_folder: Experiment folder path
            
        Returns:
            Path to saved plot
        """
        plots_dir = os.path.join(exp_folder, 'plots')
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Convert values to float (handle numpy arrays/scalars)
        def to_float(val):
            if isinstance(val, np.ndarray):
                # If array has multiple elements, take the mean
                if val.size > 1:
                    return float(np.mean(val))
                else:
                    return float(val.item())
            elif isinstance(val, np.generic):
                return float(val)
            elif isinstance(val, (int, float)):
                return float(val)
            else:
                return 0.0
        
        metric_values_float = [to_float(v) for v in metric_values]
        
        # Create bar chart
        fig = go.Figure()
        
        # Color code by metric type
        colors = []
        for name in metric_names:
            if 'MAE' in name or 'RMSE' in name:
                colors.append('#2E86AB')  # Blue for absolute errors
            elif 'MSE' in name:
                colors.append('#A23B72')  # Purple for squared errors
            elif 'MAPE' in name or 'MSPE' in name:
                colors.append('#F18F01')  # Orange for percentage errors
            elif 'CORR' in name:
                colors.append('#06A77D')  # Green for correlation
            else:
                colors.append('#D62246')  # Red for others
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values_float,
            marker_color=colors,
            text=[f'{v:.6f}' for v in metric_values_float],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=dict(text='Performance Metrics Summary', font=dict(size=18)),
            xaxis_title="Metric",
            yaxis_title="Value",
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        save_path = os.path.join(plots_dir, 'metrics_summary.html')
        fig.write_html(save_path)
        
        return save_path
    
    def create_markdown_report(self,
                              setting: str,
                              metrics: Dict[str, float],
                              exp_folder: str,
                              args: Any,
                              dtw_value: Optional[float] = None) -> str:
        """
        Create comprehensive Markdown report.
        
        Args:
            setting: Experiment setting string
            metrics: Dictionary of metrics
            exp_folder: Experiment folder path
            args: Experiment arguments
            dtw_value: DTW metric value (optional)
            
        Returns:
            Path to saved report
        """
        reports_dir = os.path.join(exp_folder, 'reports')
        report_path = os.path.join(reports_dir, 'experiment_report.md')
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create markdown content
        md_content = f"""# Experiment Report: {setting}

**Generated:** {timestamp}

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | {args.model} |
| Dataset | {args.data} |
| Features | {args.features} |
| Sequence Length | {args.seq_len} |
| Label Length | {args.label_len} |
| Prediction Length | {args.pred_len} |
| Batch Size | {args.batch_size} |
| Learning Rate | {args.learning_rate} |
| Train Epochs | {args.train_epochs} |

---

## Performance Metrics

| Metric | Value |
|--------|-------|
"""
        
        # Add metrics to table
        for metric_name, value in sorted(metrics.items()):
            if isinstance(value, float):
                md_content += f"| {metric_name} | {value:.6f} |\n"
            else:
                md_content += f"| {metric_name} | {value} |\n"
        
        if dtw_value is not None and dtw_value != 'Not calculated':
            md_content += f"| DTW | {dtw_value:.6f} |\n"
        
        md_content += """
---

## Visualizations

### Prediction Plots
- 📊 [Sample Predictions](../plots/prediction_sample_0_feature_-1.html)
- 📈 [Feature Comparison](../plots/feature_comparison.html)

### Analysis Plots
- 🔍 [Error Analysis](../plots/error_analysis.html)
- 📉 [Metrics Summary](../plots/metrics_summary.html)

### Comprehensive Report
- 🎨 [Interactive HTML Report](../comprehensive_results.html)

---

## Data Files

- `data/predictions.npy` - Full predictions array
- `data/ground_truth.npy` - Ground truth array
- `data/metrics.json` - Metrics in JSON format
- `data/metrics.npy` - Core metrics array

---

## Notes

This report was automatically generated by the Time Series Forecasting Framework.

For interactive visualizations, open the HTML files in your web browser.

"""
        
        # Write report
        with open(report_path, 'w') as f:
            f.write(md_content)
        
        return report_path
    
    def create_all_visualizations(self,
                                 preds: np.ndarray,
                                 trues: np.ndarray,
                                 inputs: Optional[np.ndarray],
                                 metrics: Dict[str, float],
                                 setting: str,
                                 args: Any,
                                 dtw_value: Optional[float] = None,
                                 num_samples: int = 3) -> str:
        """
        Create all visualizations and reports for an experiment.
        
        Args:
            preds: Predictions array
            trues: Ground truth array
            inputs: Input sequences array (optional)
            metrics: Dictionary of metrics
            setting: Experiment setting string
            args: Experiment arguments
            dtw_value: DTW metric value (optional)
            num_samples: Number of sample predictions to visualize
            
        Returns:
            Path to experiment folder
        """
        # Create folder structure
        exp_folder = self.create_experiment_folder(setting)
        
        print(f"\n{'='*80}")
        print(f"📁 Creating Comprehensive Results Package".center(80))
        print(f"{'='*80}\n")
        
        # Save raw data
        print("💾 Saving prediction data...")
        self.save_predictions(preds, trues, inputs, exp_folder)
        
        print("💾 Saving metrics...")
        self.save_metrics(metrics, exp_folder, {'setting': setting})
        
        # Create visualizations
        print("\n📊 Generating visualizations...")
        
        # Sample predictions
        n_features = preds.shape[-1]
        for sample_idx in range(min(num_samples, preds.shape[0])):
            # For multivariate, plot middle feature; for univariate, plot last
            feature_idx = n_features // 2 if n_features > 1 else -1
            self.create_prediction_plot(
                preds, trues, inputs, exp_folder, 
                sample_idx=sample_idx, 
                feature_idx=feature_idx,
                title=f"Sample {sample_idx + 1}"
            )
            print(f"  ✓ Created prediction plot for sample {sample_idx + 1}")
        
        # Error analysis
        print("  ⚡ Generating error analysis...")
        self.create_error_analysis_plot(preds, trues, exp_folder)
        
        # Feature comparison (if multivariate)
        if n_features > 1:
            print("  ⚡ Generating feature comparison...")
            self.create_feature_comparison_plot(preds, trues, exp_folder)
        
        # Metrics summary
        print("  ⚡ Generating metrics summary...")
        self.create_metrics_summary_plot(metrics, exp_folder)
        
        # Create markdown report
        print("\n📝 Creating markdown report...")
        report_path = self.create_markdown_report(
            setting, metrics, exp_folder, args, dtw_value
        )
        
        print(f"\n{'='*80}")
        print(f"✨ Results Package Complete!".center(80))
        print(f"{'='*80}")
        print(f"\n📂 Location: {exp_folder}")
        print(f"📄 Report: {report_path}")
        print(f"\n💡 Open the markdown report or HTML visualizations in your browser!\n")
        
        return exp_folder
