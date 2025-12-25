"""
Enhanced visualization utilities for time series forecasting results.
Provides interactive plots, comparison charts, and visual summaries.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class ForecastVisualizer:
    """
    Manages all visualization for time series forecasting experiments.
    Stores all predictions and creates a comprehensive HTML report.
    """
    
    def __init__(self, experiment_name: str = "Experiment"):
        """
        Initialize the visualizer.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.predictions_data = []
        self.metrics_data = []
    
    def add_prediction(
        self,
        ground_truth: np.ndarray,
        prediction: np.ndarray,
        input_seq: Optional[np.ndarray] = None,
        sample_id: str = "Sample",
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Add a prediction to the visualizer for later plotting.
        
        Args:
            ground_truth: Ground truth values
            prediction: Predicted values
            input_seq: Input sequence (optional)
            sample_id: Identifier for this sample
            metrics: Performance metrics for this sample
        """
        self.predictions_data.append({
            'ground_truth': ground_truth,
            'prediction': prediction,
            'input_seq': input_seq,
            'sample_id': sample_id,
            'metrics': metrics or {}
        })
        
        if metrics:
            metrics['sample_id'] = sample_id
            self.metrics_data.append(metrics)
    
    def create_comprehensive_html(
        self,
        save_path: str,
        max_predictions: int = 50,
        features_to_plot: Optional[List[int]] = None
    ):
        """
        Create a comprehensive HTML report with all predictions.
        
        Args:
            save_path: Path to save the HTML file
            max_predictions: Maximum number of predictions to show
            features_to_plot: List of feature indices to plot (for multivariate)
        """
        if not self.predictions_data:
            print("No predictions to visualize!")
            return
        
        # Limit number of predictions
        pred_data = self.predictions_data[:max_predictions]
        n_predictions = len(pred_data)
        
        print(f"Creating visualization for {n_predictions} time series...")
        
        # Calculate adaptive vertical spacing based on number of rows
        # Max spacing is 1/(n_predictions - 1), we use 80% of that for safety
        if n_predictions > 1:
            max_spacing = 1.0 / (n_predictions - 1)
            vertical_spacing = min(0.08, max_spacing * 0.8)
        else:
            vertical_spacing = 0.08
        
        # Create subplots
        fig = make_subplots(
            rows=n_predictions,
            cols=1,
            subplot_titles=[f"{p['sample_id']} | " + 
                          " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                     for k, v in p['metrics'].items()])
                          for p in pred_data],
            vertical_spacing=vertical_spacing,
            specs=[[{"secondary_y": True}] for _ in range(n_predictions)]
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, pred_info in enumerate(pred_data, 1):
            ground_truth = pred_info['ground_truth']
            prediction = pred_info['prediction']
            input_seq = pred_info['input_seq']
            
            # Handle multivariate by selecting first feature or specified feature
            if ground_truth.ndim > 1:
                feature_idx = features_to_plot[0] if features_to_plot else 0
                ground_truth = ground_truth[:, feature_idx]
                prediction = prediction[:, feature_idx]
                if input_seq is not None and input_seq.ndim > 1:
                    input_seq = input_seq[:, feature_idx]
            
            # Plot input sequence if available
            if input_seq is not None:
                input_len = len(input_seq)
                pred_len = len(ground_truth)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(input_len)),
                        y=input_seq,
                        mode='lines',
                        name='Input' if idx == 1 else None,
                        showlegend=(idx == 1),
                        line=dict(color=colors[0], width=1.5),
                        opacity=0.6,
                        legendgroup='input',
                        hovertemplate='Input: %{y:.4f}<extra></extra>'
                    ),
                    row=idx, col=1, secondary_y=False
                )
                
                pred_x = list(range(input_len, input_len + pred_len))
                
                # Add forecast start line
                fig.add_vline(
                    x=input_len,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.4,
                    row=idx, col=1
                )
            else:
                pred_x = list(range(len(ground_truth)))
            
            # Plot ground truth
            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=ground_truth,
                    mode='lines',
                    name='Ground Truth' if idx == 1 else None,
                    showlegend=(idx == 1),
                    line=dict(color=colors[1], width=2.5),
                    legendgroup='gt',
                    hovertemplate='GT: %{y:.4f}<extra></extra>'
                ),
                row=idx, col=1, secondary_y=False
            )
            
            # Plot prediction
            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=prediction,
                    mode='lines',
                    name='Prediction' if idx == 1 else None,
                    showlegend=(idx == 1),
                    line=dict(color=colors[2], width=2, dash='dash'),
                    legendgroup='pred',
                    hovertemplate='Pred: %{y:.4f}<extra></extra>'
                ),
                row=idx, col=1, secondary_y=False
            )
            
            # Plot error on secondary y-axis
            errors = prediction - ground_truth
            fig.add_trace(
                go.Scatter(
                    x=pred_x,
                    y=errors,
                    mode='lines',
                    name='Error' if idx == 1 else None,
                    showlegend=(idx == 1),
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    legendgroup='error',
                    hovertemplate='Error: %{y:.4f}<extra></extra>'
                ),
                row=idx, col=1, secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>{self.experiment_name}</b> - Comprehensive Results",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            height=400 * n_predictions,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        for i in range(1, n_predictions + 1):
            fig.update_xaxes(title_text="Time Step", row=i, col=1)
            fig.update_yaxes(title_text="Value", row=i, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Error", row=i, col=1, secondary_y=True)
        
        # Save as HTML
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Interactive visualization saved to: {save_path}")
        
        # Also create metrics summary if we have metrics
        if self.metrics_data:
            self._create_metrics_summary(os.path.dirname(save_path))
    
    def _create_metrics_summary(self, output_dir: str):
        """
        Create a separate metrics summary visualization.
        
        Args:
            output_dir: Directory to save the metrics summary
        """
        if not self.metrics_data:
            return
        
        df = pd.DataFrame(self.metrics_data)
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'sample_id' in df.columns:
            numeric_cols = [col for col in numeric_cols if col != 'sample_id']
        
        if not numeric_cols:
            return
        
        # Create metrics comparison plot
        fig = make_subplots(
            rows=len(numeric_cols),
            cols=1,
            subplot_titles=[f"{col.upper()}" for col in numeric_cols],
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Plotly
        
        for idx, metric in enumerate(numeric_cols, 1):
            fig.add_trace(
                go.Bar(
                    x=df['sample_id'] if 'sample_id' in df.columns else list(range(len(df))),
                    y=df[metric],
                    name=metric,
                    marker_color=colors[idx % len(colors)],
                    showlegend=False,
                    hovertemplate=f'<b>{metric}</b><br>Value: %{{y:.4f}}<extra></extra>'
                ),
                row=idx, col=1
            )
            
            # Add mean line
            mean_val = df[metric].mean()
            fig.add_hline(
                y=mean_val,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text=f"Mean: {mean_val:.4f}",
                annotation_position="right",
                row=idx, col=1
            )
        
        fig.update_layout(
            title=dict(
                text=f"<b>{self.experiment_name}</b> - Metrics Summary",
                x=0.5,
                xanchor='center'
            ),
            height=300 * len(numeric_cols),
            template='plotly_white',
            showlegend=False
        )
        
        for i in range(1, len(numeric_cols) + 1):
            fig.update_xaxes(title_text="Sample", row=i, col=1)
        
        metrics_path = os.path.join(output_dir, "metrics_summary.html")
        fig.write_html(metrics_path)
        print(f"📈 Metrics summary saved to: {metrics_path}")
