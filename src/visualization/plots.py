import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_prediction(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    input_seq: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "Time Series Prediction",
    feature_idx: int = 0,
    show_confidence: bool = False,
    confidence_std: Optional[np.ndarray] = None
) -> None:
    """
    Create an enhanced prediction visualization using Plotly.
    
    Args:
        ground_truth: Ground truth values
        prediction: Predicted values
        input_seq: Input sequence (optional)
        save_path: Path to save the figure (as HTML)
        title: Plot title
        feature_idx: Feature index to plot for multivariate series
        show_confidence: Whether to show confidence intervals
        confidence_std: Standard deviation for confidence intervals
    """
    fig = go.Figure()
    
    # Determine x-axis positions
    if input_seq is not None:
        input_len = len(input_seq)
        pred_len = len(ground_truth)
        
        # Plot input sequence
        fig.add_trace(go.Scatter(
            x=list(range(input_len)),
            y=input_seq,
            mode='lines',
            name='Input',
            line=dict(color='#2E86AB', width=2),
            opacity=0.7
        ))
        
        # Plot ground truth and prediction
        pred_range = list(range(input_len, input_len + pred_len))
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=ground_truth,
            mode='lines',
            name='Ground Truth',
            line=dict(color='#06A77D', width=2.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=prediction,
            mode='lines',
            name='Prediction',
            line=dict(color='#D62246', width=2, dash='dash')
        ))
        
        # Add vertical line separating input and prediction
        fig.add_vline(
            x=input_len,
            line_dash="dot",
            line_color="gray",
            opacity=0.6,
            annotation_text="Forecast Start",
            annotation_position="top"
        )
    else:
        pred_range = list(range(len(ground_truth)))
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=ground_truth,
            mode='lines',
            name='Ground Truth',
            line=dict(color='#06A77D', width=2.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=pred_range,
            y=prediction,
            mode='lines',
            name='Prediction',
            line=dict(color='#D62246', width=2, dash='dash')
        ))
    
    # Add confidence intervals if provided
    if show_confidence and confidence_std is not None:
        lower_bound = prediction - 2 * confidence_std
        upper_bound = prediction + 2 * confidence_std
        
        fig.add_trace(go.Scatter(
            x=pred_range + pred_range[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(214, 34, 70, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='95% Confidence'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Time Step",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        height=500
    )
    
    if save_path:
        # Convert .pdf or .png to .html
        if not save_path.endswith('.html'):
            save_path = os.path.splitext(save_path)[0] + '.html'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Visualization saved to: {save_path}")
    else:
        fig.show()


def plot_multivariate_predictions(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    max_features: int = 4
) -> None:
    """
    Plot multiple features in a grid layout using Plotly.
    
    Args:
        ground_truth: Ground truth array (seq_len, n_features)
        predictions: Predictions array (seq_len, n_features)
        feature_names: Names of features
        save_path: Path to save the figure (as HTML)
        max_features: Maximum number of features to plot
    """
    n_features = min(ground_truth.shape[-1], max_features)
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Create subplots
    fig = make_subplots(
        rows=n_features,
        cols=1,
        subplot_titles=feature_names[:n_features],
        vertical_spacing=0.08
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx in range(n_features):
        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=list(range(len(ground_truth))),
                y=ground_truth[:, idx],
                mode='lines',
                name='Ground Truth' if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=colors[0], width=2),
                legendgroup='gt'
            ),
            row=idx+1, col=1
        )
        
        # Prediction
        fig.add_trace(
            go.Scatter(
                x=list(range(len(predictions))),
                y=predictions[:, idx],
                mode='lines',
                name='Prediction' if idx == 0 else None,
                showlegend=(idx == 0),
                line=dict(color=colors[1], width=2, dash='dash'),
                legendgroup='pred'
            ),
            row=idx+1, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=400 * n_features,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update axes
    for i in range(1, n_features + 1):
        fig.update_xaxes(title_text="Time Step", row=i, col=1)
        fig.update_yaxes(title_text="Value", row=i, col=1)
    
    if save_path:
        # Convert .pdf or .png to .html
        if not save_path.endswith('.html'):
            save_path = os.path.splitext(save_path)[0] + '.html'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Visualization saved to: {save_path}")
    else:
        fig.show()


def plot_error_distribution(
    errors: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Prediction Error Distribution"
) -> None:
    """
    Plot the distribution of prediction errors using Plotly.
    
    Args:
        errors: Array of errors (predictions - ground_truth)
        save_path: Path to save the figure (as HTML)
        title: Plot title
    """
    errors_flat = errors.flatten()
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Error Histogram', 'Q-Q Plot (Normality Check)'),
        horizontal_spacing=0.12
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=errors_flat,
            nbinsx=50,
            marker_color='#2E86AB',
            opacity=0.7,
            name='Error Distribution'
        ),
        row=1, col=1
    )
    
    # Add zero line
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text="Zero Error",
        row=1, col=1
    )
    
    # Q-Q plot
    try:
        from scipy import stats
        theoretical_quantiles = stats.probplot(errors_flat, dist="norm")[0][0]
        sample_quantiles = stats.probplot(errors_flat, dist="norm")[0][1]
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                marker=dict(color='#2E86AB', size=4),
                name='Q-Q Plot'
            ),
            row=1, col=2
        )
        
        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Fit',
                showlegend=False
            ),
            row=1, col=2
        )
    except ImportError:
        # If scipy not available, show message
        fig.add_annotation(
            text="scipy not available for Q-Q plot",
            xref="x2", yref="y2",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        showlegend=True,
        template='plotly_white',
        height=500
    )
    
    fig.update_xaxes(title_text="Error", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
    
    if save_path:
        # Convert .pdf or .png to .html
        if not save_path.endswith('.html'):
            save_path = os.path.splitext(save_path)[0] + '.html'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Visualization saved to: {save_path}")
    else:
        fig.show()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = "Model Comparison"
) -> None:
    """
    Create a bar chart comparing metrics across different models/experiments using Plotly.
    
    Args:
        metrics_dict: Dictionary with format {model_name: {metric_name: value}}
        save_path: Path to save the figure (as HTML)
        title: Plot title
    """
    # Prepare data
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    # Create DataFrame for easier plotting
    data = []
    for model, metrics in metrics_dict.items():
        for metric, value in metrics.items():
            data.append({'Model': model, 'Metric': metric, 'Value': value})
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, metric in enumerate(metric_names):
        metric_data = [metrics_dict[model].get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric,
            x=models,
            y=metric_data,
            marker_color=colors[i % len(colors)],
            text=[f'{val:.4f}' for val in metric_data],
            textposition='auto'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Model",
        yaxis_title="Value",
        barmode='group',
        template='plotly_white',
        showlegend=True,
        height=500,
        hovermode='x unified'
    )
    
    if save_path:
        # Convert .pdf or .png to .html
        if not save_path.endswith('.html'):
            save_path = os.path.splitext(save_path)[0] + '.html'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Visualization saved to: {save_path}")
    else:
        fig.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> None:
    """
    Plot training history with train/validation losses using Plotly.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Path to save the figure (as HTML)
        title: Plot title
    """
    fig = go.Figure()
    
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    
    if 'train_loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['train_loss'],
            mode='lines+markers',
            name='Train Loss',
            line=dict(color='#2E86AB', width=2.5),
            marker=dict(size=6, symbol='circle')
        ))
    
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='#D62246', width=2.5),
            marker=dict(size=6, symbol='square')
        ))
    
    if 'test_loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['test_loss'],
            mode='lines+markers',
            name='Test Loss',
            line=dict(color='#06A77D', width=2.5),
            marker=dict(size=6, symbol='triangle-up')
        ))
    
    # Mark best epoch if validation loss exists
    if 'val_loss' in history:
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val = min(history['val_loss'])
        
        fig.add_vline(
            x=best_epoch,
            line_dash="dot",
            line_color="gray",
            opacity=0.6
        )
        
        fig.add_trace(go.Scatter(
            x=[best_epoch],
            y=[best_val],
            mode='markers',
            name=f'Best (Epoch {best_epoch})',
            marker=dict(size=15, color='red', symbol='star')
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=500
    )
    
    if save_path:
        # Convert .pdf or .png to .html
        if not save_path.endswith('.html'):
            save_path = os.path.splitext(save_path)[0] + '.html'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"📊 Visualization saved to: {save_path}")
    else:
        fig.show()


def create_experiment_summary_plot(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    input_seq: np.ndarray,
    metrics: Dict[str, float],
    save_path: str,
    title: str = "Experiment Summary"
) -> None:
    """
    Create a comprehensive summary plot with predictions and metrics using Plotly.
    
    Args:
        ground_truth: Ground truth values
        prediction: Predicted values
        input_seq: Input sequence
        metrics: Dictionary of metrics
        save_path: Path to save the figure (as HTML)
        title: Plot title
    """
    # Create subplots - main prediction plot and error plot
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, 'Prediction Error'),
        vertical_spacing=0.12
    )
    
    input_len = len(input_seq)
    pred_len = len(ground_truth)
    pred_range = list(range(input_len, input_len + pred_len))
    
    # Main prediction plot
    fig.add_trace(
        go.Scatter(
            x=list(range(input_len)),
            y=input_seq,
            mode='lines',
            name='Input',
            line=dict(color='#2E86AB', width=2),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_range,
            y=ground_truth,
            mode='lines',
            name='Ground Truth',
            line=dict(color='#06A77D', width=2.5)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_range,
            y=prediction,
            mode='lines',
            name='Prediction',
            line=dict(color='#D62246', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    fig.add_vline(
        x=input_len,
        line_dash="dot",
        line_color="gray",
        opacity=0.6,
        row=1, col=1
    )
    
    # Error plot
    errors = prediction - ground_truth
    fig.add_trace(
        go.Scatter(
            x=pred_range,
            y=errors,
            mode='lines',
            name='Error',
            line=dict(color='#D62246', width=2),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        row=2, col=1
    )
    
    # Create metrics text
    metrics_text = "<br>".join([
        f"<b>{k}</b>: {v:.6f}" if isinstance(v, float) else f"<b>{k}</b>: {v}"
        for k, v in metrics.items()
    ])
    
    # Add metrics annotation
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=1.02, y=0.5,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 224, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10, family="monospace")
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        margin=dict(r=200)  # Extra margin for metrics
    )
    
    fig.update_xaxes(title_text="Time Step", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    
    # Convert .pdf or .png to .html
    if not save_path.endswith('.html'):
        save_path = os.path.splitext(save_path)[0] + '.html'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"📊 Experiment summary saved to: {save_path}")


def visual(true, preds=None, name='./pic/test.html'):
    """
    Legacy visualization function - now uses Plotly instead of matplotlib.
    Results visualization with improved aesthetics.
    
    Args:
        true: Ground truth values
        preds: Predicted values (optional)
        name: Path to save the figure (will be saved as HTML)
    """
    fig = go.Figure()
    
    if preds is not None:
        fig.add_trace(go.Scatter(
            x=list(range(len(preds))),
            y=preds,
            mode='lines',
            name='Prediction',
            line=dict(color='#D62246', width=2.5, dash='dash')
        ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(true))),
        y=true,
        mode='lines',
        name='Ground Truth',
        line=dict(color='#06A77D', width=2.5)
    ))
    
    fig.update_layout(
        title=dict(text='Time Series Prediction', font=dict(size=18)),
        xaxis_title="Time Step",
        yaxis_title="Value",
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        height=500
    )
    
    # Ensure filename has .html extension
    if not name.endswith('.html'):
        name = os.path.splitext(name)[0] + '.html'
    
    os.makedirs(os.path.dirname(name), exist_ok=True)
    fig.write_html(name)
    print(f"📊 Visualization saved to: {name}")
