"""
Time Series Forecasting Experiment Implementation
=============================================================

This module implements the complete pipeline for time series forecasting experiments.
It handles the entire lifecycle of a forecasting model including:
- Model initialization and configuration
- Training with various optimization strategies
- Validation for model selection
- Testing and comprehensive evaluation
- Visualization and results reporting

The implementation supports multiple model architectures (RAFT, TimeMixer, WPMixer)
and various forecasting configurations (univariate, multivariate, with/without retrieval etc.).

Key Features:
-------------
1. **Flexible Model Support**: Works with retrieval-based and non-retrieval models
2. **Multiple Forecasting Modes**: Univariate (S), Multivariate (M), Multi-to-Single (MS)
3. **Comprehensive Evaluation**: Standard metrics (MAE, MSE, RMSE) + optional DTW
4. **Rich Visualizations**: Interactive HTML reports with prediction plots
5. **Early Stopping**: Prevents overfitting during training
6. **Mixed Precision Training**: Optional AMP for faster training on GPUs

Architecture:
-------------
The experiment follows the template method pattern:
    Exp_Basic (abstract base)
        └── Exp_Forecast (this class)
            ├── _build_model()      : Initialize model architecture
            ├── _get_data()         : Load and prepare datasets
            ├── _select_optimizer() : Configure optimization algorithm
            ├── _select_criterion() : Define loss function
            ├── train()             : Main training loop
            ├── vali()              : Validation during training
            └── test()              : Final evaluation and visualization

Typical Usage:
--------------
    args = parse_args()  # Configuration from command line
    exp = Exp_Forecast(args)
    
    # Train the model
    setting = f'{args.model}_{args.data}_pl{args.pred_len}'
    model = exp.train(setting)
    
    # Evaluate on test set
    exp.test(setting, test=0)

Author: Time Series Library Team
Last Modified: 2025
"""

# Standard library imports
import os
import time
import warnings

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# Local application imports
from src.data import data_provider
from .base import Exp_Basic
from src.utils.training_utils import EarlyStopping, adjust_learning_rate
from src.utils import ResultsManager
from src.visualization import visual, ForecastVisualizer
from src.evaluation import metric, calculate_all_metrics, format_metrics
from src.evaluation import dtw, accelerated_dtw

# Suppress warnings for cleaner output during experiments
warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    """
    Time Series Forecasting Experiment Class
    =========================================

    This class orchestrates the complete forecasting experiment pipeline, extending
    the base experiment class (Exp_Basic) with forecasting-specific functionality.
    
    The class handles three main phases:
    1. **Training Phase**: Iterative model optimization with validation monitoring
    2. **Validation Phase**: Model selection based on validation set performance
    3. **Testing Phase**: Final evaluation with comprehensive metrics and visualizations
    
    Supported Model Types:
    ----------------------
    - **Retrieval-based Models**: Models that use historical pattern retrieval (RAFT)
      These models can leverage a database of past sequences to improve predictions.
    
    - **Non-retrieval Models**: Standard sequence-to-sequence architectures (TimeMixer, etc.)
      These models rely solely on learned patterns without explicit retrieval.
    
    Key Attributes:
    ---------------
    - args: Configuration object containing all hyperparameters and settings
    - model: The neural network model being trained/evaluated
    - device: Computation device (CPU/GPU)
    - model_dict: Dictionary mapping model names to their classes
    
    Configuration Parameters (from args):
    -------------------------------------
    - model: Model architecture name ('RAFT', 'TimeMixer', 'WPMixer')
    - data: Dataset name ('ETTh1', 'ETTm1', 'pedestrian', etc.)
    - features: Forecasting mode ('S'=univariate, 'M'=multivariate, 'MS'=multi-to-single)
    - seq_len: Input sequence length (lookback window)
    - label_len: Decoder input length (for attention models)
    - pred_len: Prediction horizon (forecast length)
    - enc_in: Number of encoder input features
    - use_retrieval: Whether to use retrieval-based forecasting
    - use_amp: Whether to use automatic mixed precision training
    - train_epochs: Maximum number of training epochs
    - patience: Early stopping patience (epochs without improvement)
    - learning_rate: Initial learning rate for optimizer
    - inverse: Whether to inverse-transform predictions back to original scale
    - use_dtw: Whether to calculate Dynamic Time Warping metric
    """
    
    def __init__(self, args):
        """
        Initialize the time series forecasting experiment.
        
        This constructor performs the following steps:
        1. Calls parent class constructor to set up base experiment infrastructure
        2. Inherits model dictionary, device configuration, and logging setup
        3. Prepares the experiment for training/testing phases
        
        The actual model initialization happens lazily in _build_model() when needed,
        allowing for proper device placement and data preparation for retrieval models.
        
        Parameters:
        -----------
        args : argparse.Namespace or object
            Configuration object containing all experiment settings including:
            - Model architecture and hyperparameters
            - Dataset specifications and paths
            - Training configuration (epochs, learning rate, batch size)
            - Hardware settings (GPU usage, mixed precision)
            - Evaluation options (metrics, visualization)
        
        Side Effects:
        -------------
        - Sets up device (CPU/GPU) for tensor operations
        - Initializes model dictionary for architecture selection
        - Prepares logging infrastructure
        """
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        """
        Build and initialize the forecasting model architecture.
        
        This method performs several critical initialization steps:
        
        1. **Model Instantiation**: Creates the model from the model dictionary
           using the specified architecture name (e.g.,'RAFT')
        
        2. **Multi-GPU Setup**: If multiple GPUs are available and enabled,
           wraps the model with DataParallel for distributed training
        
        3. **Retrieval Dataset Preparation**: For retrieval-based models (RAFT),
           prepares the retrieval database from train/val/test datasets. This allows
           the model to search and retrieve similar historical patterns during inference.
        
        Model Selection Process:
        -------------------------
        The model is selected from self.model_dict using self.args.model as key.
        Each model class must implement a consistent interface accepting args and
        returning predictions with shape [batch_size, pred_len, features].
        
        Retrieval Model Initialization:
        --------------------------------
        Retrieval-based models need access to all data splits to build their
        pattern database. The preparation involves:
        - Loading train, validation, and test datasets
        - Moving model to device before dataset preparation
        - Calling model.prepare_dataset() to index historical patterns
        
        This enables the model to perform k-nearest neighbor retrieval during
        forecasting, finding similar past sequences to inform predictions.
        
        Multi-GPU Training:
        -------------------
        When use_multi_gpu=True and multiple GPUs are available:
        - Model is wrapped with nn.DataParallel
        - Data batches are automatically split across GPUs
        - Gradients are synchronized during backward pass
        - This provides near-linear speedup for large batch sizes
        
        Returns:
        --------
        torch.nn.Module
            The initialized model ready for training or evaluation.
            The model is in training mode by default.
        
        Side Effects:
        -------------
        - Model is moved to self.device (CPU/GPU)
        - For retrieval models, internal retrieval database is populated
        - Model parameters are initialized according to each architecture's defaults
        
        Example Model Architectures:
        ----------------------------
        - RAFT: Retrieval with adaptive feature transformation
        - TimeMixer: Multi-scale temporal mixing without retrieval
        - WPMixer: Wavelet-based mixing architecture
        """
        # Instantiate model from the model dictionary using configuration
        model = self.model_dict[self.args.model](self.args).float()

        # Enable data parallelism for multi-GPU training if requested
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        # Special initialization for retrieval-based models (RAFT)
        # These models need access to the full dataset to build their retrieval database
        if self.args.use_retrieval:
            # Load all three data splits for retrieval database construction
            train_data, train_loader = self._get_data(flag='train')
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')
            
            # Move model to device before dataset preparation
            # This ensures retrieval embeddings are computed on the correct device
            model.to(self.device)
            
            # Prepare the retrieval database with all available data
            # This indexes historical patterns for k-NN retrieval during inference
            model.prepare_dataset(train_data, vali_data, test_data)
        
        return model

    def _get_data(self, flag):
        """
        Load and prepare the dataset for a specific split (train/validation/test).
        
        This method acts as a factory function that creates both the dataset object
        and its corresponding DataLoader, properly configured for the specified split.
        
        Dataset Splits:
        ---------------
        - 'train': Training set for model optimization
                   - Typically 60-70% of total data
                   - Used for gradient updates
                   - May include data augmentation
        
        - 'val': Validation set for hyperparameter tuning and model selection
                 - Typically 10-20% of total data
                 - Used for early stopping and checkpoint selection
                 - Never used for gradient updates
        
        - 'test': Test set for final unbiased evaluation
                  - Typically 10-20% of total data
                  - Used only after all training decisions are finalized
                  - Provides generalization performance estimate
        
        Data Pipeline:
        --------------
        The data_provider function (from src.data.factory) handles:
        1. Loading raw data from CSV files
        2. Splitting into appropriate time ranges based on flag
        3. Feature scaling/normalization (StandardScaler by default)
        4. Creating sliding windows (seq_len → pred_len)
        5. Wrapping in PyTorch DataLoader with proper batching
        
        Data Format:
        ------------
        Each batch from the DataLoader contains:
        - batch_x: Input sequences [batch_size, seq_len, features]
        - batch_y: Target sequences [batch_size, label_len + pred_len, features]
        - batch_x_mark: Time stamps for input [batch_size, seq_len, time_features]
        - batch_y_mark: Time stamps for output [batch_size, label_len + pred_len, time_features]
        - index (retrieval models only): Sample indices for retrieval [batch_size]
        
        Parameters:
        -----------
        flag : str
            Dataset split identifier. Must be one of:
            - 'train': Training data
            - 'val': Validation data  
            - 'test': Test data
        
        Returns:
        --------
        tuple of (Dataset, DataLoader)
            - Dataset: The underlying dataset object with access to:
                * data: Raw numpy arrays
                * scaler: Fitted StandardScaler for inverse transform
                * inverse_transform(): Method to convert predictions back to original scale
            
            - DataLoader: PyTorch DataLoader providing batched iteration with:
                * Proper shuffling (enabled for train, disabled for val/test)
                * Configured batch size
                * Number of workers for parallel data loading
                * Pin memory for faster GPU transfer (if applicable)
        
        Example Usage:
        --------------
        >>> train_data, train_loader = self._get_data(flag='train')
        >>> for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
        >>>     # batch_x: [32, 96, 7]  (batch_size=32, seq_len=96, features=7)
        >>>     # batch_y: [32, 144, 7] (label_len=48 + pred_len=96 = 144)
        >>>     predictions = model(batch_x, batch_x_mark, ...)
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Initialize and configure the optimization algorithm.
        
        This method creates the optimizer responsible for updating model parameters
        during training. The optimizer implements gradient-based learning to minimize
        the loss function.
        
        Optimizer Choice: Adam
        ----------------------
        We use Adam (Adaptive Moment Estimation) optimizer because:
        
        1. **Adaptive Learning Rates**: Maintains per-parameter learning rates that
           adapt based on gradient history, making it robust to hyperparameter choices
        
        2. **Momentum**: Incorporates first and second moment estimates of gradients,
           helping to accelerate convergence and escape local minima
        
        3. **Effective for Time Series**: Proven performance on sequence modeling tasks
           with non-stationary objectives (common in time series forecasting)
        
        4. **Computational Efficiency**: Similar cost to vanilla SGD but with better
           convergence properties in practice
        
        Learning Rate Strategy:
        -----------------------
        - Initial learning rate is set from self.args.learning_rate
        - Learning rate is adjusted during training via adjust_learning_rate()
        - Common strategy: Reduce learning rate when validation loss plateaus
        - This allows fine-tuning in later epochs for better convergence
        
        Alternative Optimizers (not currently used):
        ---------------------------------------------
        - SGD: Simpler but requires more careful tuning
        - AdamW: Adam with decoupled weight decay (good for large models)
        - RMSprop: Similar to Adam but simpler (no bias correction)
        - LAMB: Layer-wise adaptive rate scaling (for very large batch sizes)
        
        Returns:
        --------
        torch.optim.Optimizer
            Configured Adam optimizer with:
            - params: All trainable parameters from self.model
            - lr: Initial learning rate from configuration
            - betas: (0.9, 0.999) default momentum coefficients
            - eps: 1e-8 for numerical stability
        
        Side Effects:
        -------------
        - Optimizer maintains internal state (momentum buffers) for each parameter
        - Memory usage increases proportionally to model size
        
        Example:
        --------
        >>> optimizer = self._select_optimizer()
        >>> loss.backward()  # Compute gradients
        >>> optimizer.step()  # Update parameters
        >>> optimizer.zero_grad()  # Clear gradients for next iteration
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Select and initialize the loss function (criterion) for training.
        
        Loss Function: Mean Squared Error (MSE)
        ----------------------------------------
        We use MSE as the primary training objective because:
        
        1. **Smooth Gradients**: MSE is differentiable everywhere, providing stable
           gradients for backpropagation without discontinuities
        
        2. **Penalizes Large Errors**: Quadratic penalty means the model focuses more
           on reducing large prediction errors than small ones, which is often desirable
        
        3. **Gaussian Assumption**: MSE is the maximum likelihood estimator when errors
           are assumed to be normally distributed, a reasonable assumption for many
           time series forecasting problems
        
        4. **Standard Benchmark**: Widely used in time series literature, making results
           comparable across different papers and implementations
        
        Mathematical Definition:
        ------------------------
        For predictions ŷ and ground truth y:
        
            MSE = (1/n) * Σ(ŷᵢ - yᵢ)²
        
        Where n is the total number of prediction points (batch_size × pred_len × features)
        
        Loss Computation Strategy:
        --------------------------
        - Computed element-wise across all dimensions
        - Averaged over batch, time steps, and features
        - For multi-variate forecasting, all features contribute equally
        - Can be modified with feature-specific weights if needed
        
        Alternative Loss Functions (not currently used):
        ------------------------------------------------
        - MAE (L1 Loss): More robust to outliers, but non-smooth at zero
        - Huber Loss: Combines MSE and MAE benefits, smooth and robust
        - Quantile Loss: For probabilistic forecasting at different quantiles
        - MAPE Loss: Percentage-based, but problematic with near-zero values
        
        Returns:
        --------
        torch.nn.Module
            MSE loss function that can be called as:
            loss = criterion(predictions, targets)
            
            Input shapes:
            - predictions: [batch_size, pred_len, features]
            - targets: [batch_size, pred_len, features]
            
            Output:
            - Scalar tensor representing mean squared error
        
        Example Usage:
        --------------
        >>> criterion = self._select_criterion()
        >>> predictions = model(batch_x, ...)  # [32, 96, 7]
        >>> targets = batch_y[:, -pred_len:, :]  # [32, 96, 7]
        >>> loss = criterion(predictions, targets)  # scalar
        >>> loss.backward()  # compute gradients
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Perform validation to monitor model performance during training.
        
        This method evaluates the model on the validation set without updating parameters.
        It's called after each training epoch to:
        1. Monitor training progress and detect overfitting
        2. Select the best model checkpoint
        3. Trigger early stopping if performance degrades
        
        Validation Process:
        -------------------
        Unlike training, validation:
        - Disables gradient computation (torch.no_grad) for efficiency
        - Sets model to eval mode to disable dropout/batch norm training behavior
        - Processes the entire validation set without parameter updates
        - Returns average loss across all validation samples
        
        Why Validation is Important:
        -----------------------------
        1. **Overfitting Detection**: Training loss may decrease while validation loss
           increases, indicating the model is memorizing rather than learning
        
        2. **Model Selection**: The checkpoint with lowest validation loss is typically
           the best model, not the one from the final epoch
        
        3. **Hyperparameter Tuning**: Validation performance guides decisions about
           learning rate, model capacity, regularization, etc.
        
        4. **Early Stopping**: Prevents wasting computation on epochs that don't improve
           generalization performance
        
        Data Flow:
        ----------
        For each validation batch:
        1. Load batch (input, target, time features)
        2. Prepare decoder input (label_len context + zeros for prediction)
        3. Forward pass through model (different handling for retrieval models)
        4. Extract predictions for the forecast horizon
        5. Compute loss against ground truth
        6. Accumulate losses across all batches
        
        Model-Specific Handling:
        ------------------------
        
        **Retrieval Models** (use_retrieval=True):
        - Pass sample indices for k-NN retrieval
        - Use mode='valid' to enable validation-specific behavior
        
        **Standard Models**:
        - Use standard encoder-decoder architecture
        - May use mixed precision (AMP) if enabled
        - Decoder gets context + zero-filled prediction slots
        
        Feature Dimension Handling:
        ---------------------------
        - 'MS' mode (Multi-to-Single): Predicts only last feature (f_dim=-1)
        - 'M' or 'S' modes: Predicts all features (f_dim=0)
        This ensures predictions match the specified forecasting task.
        
        Parameters:
        -----------
        vali_data : Dataset
            Validation dataset object (contains scaler, timestamps, etc.)
        
        vali_loader : DataLoader
            PyTorch DataLoader yielding validation batches
        
        criterion : torch.nn.Module
            Loss function (typically MSE) to compute validation error
        
        Returns:
        --------
        float
            Average validation loss across all batches.
            Lower values indicate better generalization performance.
        
        Side Effects:
        -------------
        - Temporarily sets model to eval mode (restored to train mode at end)
        - No parameter updates occur
        - No gradient computation (memory efficient)
        
        Example:
        --------
        >>> criterion = nn.MSELoss()
        >>> val_loss = self.vali(vali_data, vali_loader, criterion)
        >>> print(f"Validation Loss: {val_loss:.4f}")
        >>> if val_loss < best_loss:
        >>>     torch.save(model.state_dict(), 'best_model.pth')
        """
        total_loss = []
        
        # Set model to evaluation mode
        # This disables dropout and sets batch norm to use running statistics
        self.model.eval()
        
        # Disable gradient computation for efficiency and memory savings
        with torch.no_grad():
            for i, data_batch in enumerate(vali_loader):
                # Handle different batch formats between retrieval and non-retrieval models
                # Retrieval models include sample indices for k-NN search
                is_retrieval_model = len(data_batch) == 5
                if is_retrieval_model:
                    index, batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    index = None
                
                # Move data to appropriate device (CPU/GPU)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input for transformer-based models
                # Structure: [context from ground truth (label_len) | zeros for prediction (pred_len)]
                # This provides the decoder with historical context while forcing it to predict the future
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass - different handling based on model type
                if self.args.use_retrieval:
                    # Standard retrieval models (e.g., RAFT)
                    outputs = self.model(batch_x, index, mode='valid')
                else:
                    # Non-retrieval models (standard encoder-decoder)
                    if self.args.use_amp:
                        # Use automatic mixed precision for faster computation
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # Standard float32 computation
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Extract predictions for the forecast horizon only
                # Feature dimension selection depends on forecasting mode:
                # - 'MS' mode: Only predict last feature (f_dim=-1)
                # - 'M' or 'S' mode: Predict all features (f_dim=0)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Move to CPU for loss computation (avoids GPU memory buildup)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                # Compute loss for this batch
                loss = criterion(pred, true)
                total_loss.append(loss)
        
        # Calculate average loss across all validation batches
        total_loss = np.average(total_loss)
        
        # Restore model to training mode for subsequent training epochs
        self.model.train()
        
        return total_loss

    def train(self, setting):
        """
        Execute the complete training procedure for the forecasting model.
        
        This is the main training orchestration method that implements the full training loop
        with validation monitoring, early stopping, and checkpoint management.
        
        Training Pipeline Overview:
        ---------------------------
        1. **Setup Phase**:
           - Load train, validation, and test datasets
           - Initialize optimizer and loss criterion
           - Create checkpoint directory
           - Setup early stopping or best model tracking
        
        2. **Training Loop** (for each epoch):
           - Iterate through training batches
           - Forward pass: Compute predictions
           - Loss computation: Calculate error (MSE)
           - Backward pass: Compute gradients
           - Parameter update: Apply optimizer step
           - Validation: Evaluate on validation set
           - Checkpoint: Save model if validation improves
           - Learning rate adjustment: Decay or schedule learning rate
        
        3. **Finalization**:
           - Load best checkpoint based on validation performance
           - Return trained model
        
        Training Strategies by Model Type:
        -----------------------------------
        
        **Retrieval-Based Models** (RAFT):
        - Explicitly track best validation loss
        - Save checkpoint when validation improves
        - No early stopping (retrieval models are more stable)
        
        **Non-Retrieval Models** (TimeMixer, WPMixer):
        - Use early stopping with patience
        - Stop if validation doesn't improve for N epochs
        - Prevents overfitting and saves computation
        
        Loss Computation Details:
        -------------------------
        
        For standard models:
            Loss = MSE(predictions, targets)
        
        Decoder Input Construction:
        ---------------------------
        Transformer-based models need decoder input structured as:
        
            dec_inp = [context | prediction_slots]
        
        Where:
        - context: Last label_len steps from ground truth (provides historical context)
        - prediction_slots: Zero-filled tensor of length pred_len (model must predict)
        
        This teacher-forcing strategy helps the decoder learn to attend to relevant
        encoder states while generating the forecast.
        
        Mixed Precision Training (AMP):
        -------------------------------
        When use_amp=True:
        - Forward pass in float16 (faster, less memory)
        - Loss scaling to prevent underflow
        - Backward pass with scaled gradients
        - Automatic gradient unscaling before optimizer step
        
        Benefits: ~2x speedup on modern GPUs (V100, A100) with minimal accuracy loss
        
        Learning Rate Scheduling:
        -------------------------
        Learning rate is adjusted via adjust_learning_rate() based on epoch number.
        Common strategies:
        - Cosine annealing: Smooth decay following cosine curve
        - Step decay: Reduce by factor every N epochs
        - Plateau-based: Reduce when validation plateaus (not currently used)
        
        Checkpoint Management:
        ----------------------
        Models are saved to: ./checkpoints/{setting}/checkpoint.pth
        
        Where setting typically includes:
        - Model name (e.g., 'RAFT')
        - Dataset name (e.g., 'ETTh1')
        - Prediction length (e.g., 'pl96')
        
        Example: './checkpoints/RAFT_ETTh1_pl96/checkpoint.pth'
        
        Progress Visualization:
        -----------------------
        Training progress shown with tqdm progress bars displaying:
        - Current epoch/total epochs
        - Batch iteration progress
        - Current batch loss
        - Time elapsed
        
        After each epoch:
        - Train loss (average across all batches)
        - Validation loss
        - Test loss (for monitoring, not used for decisions)
        - Epoch duration
        
        Parameters:
        -----------
        setting : str
            Experiment identifier string for checkpoint management.
            Typically formatted as: f"{model}_{dataset}_pl{pred_len}"
            Example: "RAFT_ETTh1_pl96"
        
        Returns:
        --------
        torch.nn.Module
            The trained model with the best validation performance.
            Model is loaded from the checkpoint with lowest validation loss.
        
        Side Effects:
        -------------
        - Creates checkpoint directory if it doesn't exist
        - Saves model checkpoint(s) during training
        - Prints training progress and metrics to console
        - Updates model parameters in-place
        
        Example Usage:
        --------------
        >>> exp = Exp_Forecast(args)
        >>> setting = f"{args.model}_{args.data}_pl{args.pred_len}"
        >>> trained_model = exp.train(setting)
        >>> # Model with best validation performance is now ready for testing
        
        Performance Tips:
        -----------------
        1. Use larger batch sizes with AMP for better GPU utilization
        2. Enable multi-GPU training for large models/datasets
        3. Adjust patience based on dataset size (larger = more patience)
        4. Monitor train vs validation loss to detect overfitting
        5. If validation loss plateaus early, consider:
           - Increasing model capacity
           - Adding regularization (dropout, weight decay)
           - Collecting more training data
           - Tuning learning rate schedule
        """
        # ========================================================================
        # SETUP PHASE: Initialize training infrastructure
        # ========================================================================
        
        # Load all three dataset splits
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create checkpoint directory for saving model states
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        
        # Initialize early stopping for non-retrieval models
        # Early stopping prevents overfitting by monitoring validation loss
        if not self.args.use_retrieval:
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Track best validation loss for retrieval models (no early stopping)
        best_valid_loss = float('inf')

        # Initialize optimizer and loss criterion
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Initialize automatic mixed precision scaler for faster training
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # ========================================================================
        # TRAINING LOOP: Iterate through epochs
        # ========================================================================
        
        print(f"\n{'='*80}\nStarting Training: {setting}\n{'='*80}")
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            # Lists to track loss components
            data_losses = []

            # Set model to training mode (enables dropout, batch norm training)
            self.model.train()
            epoch_time = time.time()
            
            # Create progress bar for batch iteration
            pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                       desc=f'Epoch {epoch+1}/{self.args.train_epochs}',
                       ncols=100, leave=True)
            
            # ====================================================================
            # BATCH ITERATION: Process each training batch
            # ====================================================================
            
            for i, data_batch in pbar:
                # Handle different batch formats (retrieval vs non-retrieval)
                is_retrieval_model = len(data_batch) == 5
                if is_retrieval_model:
                    index, batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    index = None
                
                iter_count += 1
                
                # Zero gradients from previous iteration
                model_optim.zero_grad()
                
                # Move batch data to computation device (CPU/GPU)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input: [ground truth context | zero predictions]
                # This teacher-forcing approach helps training stability
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Determine feature dimension to predict based on forecasting mode
                f_dim = -1 if self.args.features == 'MS' else 0
                target_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = 0
                
                # ================================================================
                # FORWARD PASS: Model-specific prediction computation
                # ================================================================
                
                if self.args.use_retrieval:
                    # --- Retrieval-Based Models (RAFT) ---
                    # Standard retrieval model forward pass
                    outputs = self.model(batch_x, index, mode='train')
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, target_y)
                else:
                    # --- Non-Retrieval Models (TimeMixer, WPMixer, etc.) ---
                    if self.args.use_amp:
                        # Use automatic mixed precision for faster computation
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # Standard float32 computation
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, target_y)

                # Store batch loss for epoch averaging
                train_loss.append(loss.item())
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})

                # ================================================================
                # BACKWARD PASS: Compute gradients and update parameters
                # ================================================================
                
                if self.args.use_amp:
                    # Mixed precision backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # Standard backward pass
                    loss.backward()
                    model_optim.step()

            # Close progress bar for this epoch
            pbar.close()
            
            # Calculate epoch duration
            epoch_duration = time.time() - epoch_time
            
            # ====================================================================
            # EPOCH SUMMARY: Compute and display epoch-level metrics
            # ====================================================================
            # Compute average losses across all batches
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Display epoch summary
            print(f"\n  Epoch {epoch + 1} Summary:")
            print(f"    Train Loss: {train_loss:.6f}")
            print(f"    Valid Loss: {vali_loss:.6f}")
            print(f"    Test Loss:  {test_loss:.6f}")
            print(f"    Duration:   {epoch_duration:.2f}s")

            # Adjust learning rate based on schedule
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            # ====================================================================
            # MODEL SELECTION: Save checkpoints and handle early stopping
            # ====================================================================
            
            if self.args.use_retrieval:
                # For retrieval models: Save when validation improves
                if vali_loss < best_valid_loss:
                    best_valid_loss = vali_loss
                    torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                    print(f"  ★ Validation loss improved to {vali_loss:.6f}. Model saved!")
            else:
                # For non-retrieval models: Use early stopping
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("\n  Early stopping triggered")
                    break

        # ========================================================================
        # FINALIZATION: Load best model and return
        # ========================================================================
        
        # Load the checkpoint with best validation performance
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        print(f"\n{'='*80}\nTraining Completed\n{'='*80}\n")
        
        return self.model

    def test(self, setting, test=0):
        """
        Evaluate the trained model on the test set and generate comprehensive visualizations.
        
        This method performs the final evaluation phase after training, providing:
        1. Quantitative metrics (MAE, MSE, RMSE, MAPE, etc.)
        2. Optional DTW (Dynamic Time Warping) analysis
        3. Interactive HTML visualizations with prediction plots
        4. Result storage and management
        
        Test Phase vs Validation:
        -------------------------
        - **Validation**: Used during training for model selection (sees data during training)
        - **Test**: Used only ONCE after all decisions are made (truly unseen data)
        
        The test set provides an unbiased estimate of model performance on future data.
        It should never influence any training decisions to avoid data leakage.
        
        Evaluation Pipeline:
        --------------------
        1. **Data Loading**: Load test dataset (or reload from checkpoint)
        2. **Inference**: Generate predictions for all test samples
        3. **Inverse Scaling**: Transform predictions back to original scale
        4. **Metric Calculation**: Compute comprehensive evaluation metrics
        5. **DTW Analysis**: (Optional) Calculate Dynamic Time Warping distance
        6. **Visualization**: Create interactive plots and HTML reports
        7. **Result Storage**: Save predictions, metrics, and visualizations
        
        Inference Process:
        ------------------
        For each test batch:
        1. Load input sequence and ground truth
        2. Forward pass through model (no gradient computation)
        3. Extract predictions for forecast horizon
        4. Inverse transform if data was scaled
        5. Store predictions and ground truth for later analysis
        
        Inverse Transformation:
        -----------------------
        During training, data is typically standardized (zero mean, unit variance):
        
            x_scaled = (x - μ) / σ
        
        Inverse transformation recovers original scale:
        
            x_original = x_scaled × σ + μ
        
        This is crucial because:
        - Training on scaled data improves optimization stability
        - Evaluation must be on original scale for interpretability
        - Metrics like MAPE are scale-dependent
        
        Forecasting Modes and Feature Handling:
        ----------------------------------------
        
        **Univariate (S)**: Single time series
        - enc_in = 1, output features = 1
        - Straightforward prediction of one variable
        
        **Multivariate (M)**: Multiple related time series
        - enc_in = N, output features = N
        - Jointly predicts all N variables
        - Captures cross-variable dependencies
        
        **Multi-to-Single (MS)**: Multiple inputs, single output
        - enc_in = N, output features = 1
        - Uses N variables as context to predict one target
        - Useful when auxiliary variables help prediction
        
        Dynamic Time Warping (DTW):
        ---------------------------
        DTW measures similarity between time series that may vary in speed:
        
        - Standard metrics (MSE, MAE) require point-wise alignment
        - DTW allows flexible alignment (stretching/compressing)
        - Useful for: phase-shifted predictions, varying-speed patterns
        - Computationally expensive: O(n²) per sample
        
        When to use DTW:
        - Seasonal patterns with variable phase
        - Event detection with timing uncertainty
        - Shape similarity more important than exact alignment
        
        Visualization Strategy:
        -----------------------
        Two types of visualizations are generated:
        
        1. **Per-Sample Plots** (via ResultsManager):
           - Shows input sequence + prediction + ground truth
           - Individual PNG files for quick inspection
           - Limited to a few representative samples
        
        2. **Comprehensive HTML Report** (via ForecastVisualizer):
           - Interactive plots with zoom/pan
           - Displays predictions for multiple samples/features
           - Per-sample metrics embedded in plots
           - Easy sharing and presentation
        
        For multivariate forecasting (M/MS with multiple features):
        - Collects samples from each feature/time series
        - Visualizes each feature separately for clarity
        - Example: 7 features → 7 sets of plots (3 samples each)
        
        Result Storage Structure:
        -------------------------
        Results are saved to: ./results/{setting}/
        
        Contents:
        - predictions.npy: All model predictions [n_samples, pred_len, features]
        - ground_truth.npy: True values [n_samples, pred_len, features]
        - metrics.json: Comprehensive evaluation metrics
        - comprehensive_results.html: Interactive visualization report
        - sample_*.png: Individual prediction plots
        
        Metrics Computed:
        -----------------
        Standard regression metrics:
        - MAE: Mean Absolute Error - average absolute difference
        - MSE: Mean Squared Error - average squared difference  
        - RMSE: Root MSE - MSE in original units
        - MAPE: Mean Absolute Percentage Error - relative error
        - MSPE: Mean Squared Percentage Error
        
        Optional metrics:
        - DTW: Dynamic Time Warping distance - shape similarity
        - Per-feature metrics: Individual metrics for each variable
        
        Parameters:
        -----------
        setting : str
            Experiment identifier string for result management.
            Format: f"{model}_{dataset}_pl{pred_len}"
            Example: "RAFT_ETTh1_pl96"
        
        test : int, optional (default=0)
            Whether to load model from checkpoint:
            - 0: Use current model state (model already in memory)
            - 1: Load from checkpoint file (e.g., after training separately)
        
        Returns:
        --------
        None
            Results are saved to disk. Metrics printed to console.
        
        Side Effects:
        -------------
        - Creates results directory: ./results/{setting}/
        - Saves prediction arrays, metrics, and visualizations
        - Prints comprehensive metric summary to console
        - Generates HTML report for sharing/presentation
        
        Example Usage:
        --------------
        After training:
        >>> exp = Exp_Forecast(args)
        >>> setting = f"{args.model}_{args.data}_pl{args.pred_len}"
        >>> trained_model = exp.train(setting)
        >>> exp.test(setting, test=0)  # Evaluate trained model
        
        Loading saved model:
        >>> exp = Exp_Forecast(args)
        >>> setting = "RAFT_ETTh1_pl96"
        >>> exp.test(setting, test=1)  # Load checkpoint first
        
        Interpretation Guidelines:
        --------------------------
        **Good Performance Indicators**:
        - RMSE close to data standard deviation
        - MAPE < 10% (for most applications)
        - Visual predictions follow ground truth trends
        - DTW distance small relative to sequence length
        
        **Poor Performance Indicators**:
        - Flat predictions (model predicting mean)
        - Phase-shifted patterns (timing errors)
        - Growing error over horizon (instability)
        - High variance in per-sample errors
        
        **Common Issues**:
        - Overfitting: Good train metrics, poor test metrics
        - Underfitting: Poor metrics across all sets
        - Scale mismatch: Forgetting inverse transformation
        - Mode collapse: All predictions similar regardless of input
        
        Performance Optimization:
        -------------------------
        - DTW calculation can be slow for large test sets
        - Consider sampling subset for DTW analysis
        - Use GPU for faster inference (automatic if available)
        - Batch size can be larger than training (no backprop memory)
        """
        # ========================================================================
        # SETUP PHASE: Load data and model
        # ========================================================================
        
        test_data, test_loader = self._get_data(flag='test')
        
        # Load model from checkpoint if requested
        if test:
            print('Loading model from checkpoint...')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        # Initialize storage for predictions and ground truth
        preds = []
        trues = []
        inputs = []

        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize visualizer for comprehensive HTML report
        visualizer = ForecastVisualizer(
            experiment_name=f"{self.args.model}_{self.args.data}"
        )
        
        print(f"\n{'='*80}\nRunning Test Phase\n{'='*80}")
        
        # ========================================================================
        # VISUALIZATION SAMPLE SELECTION STRATEGY
        # ========================================================================
        
        # Determine how many features/time series to visualize
        # For multivariate (M/MS), visualize each feature separately
        # For univariate (S), visualize the single feature
        n_features_to_visualize = (
            self.args.enc_in if self.args.features in ['M', 'MS'] else 1
        )
        
        # Track how many samples collected per feature for visualization
        feature_samples_collected = [0] * n_features_to_visualize
        max_samples_per_feature = 3  # Collect a few representative samples per feature
        
        # ========================================================================
        # INFERENCE LOOP: Generate predictions for all test samples
        # ========================================================================
        
        with torch.no_grad():
            # Progress bar for test inference
            pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                       desc='Testing', ncols=100)
            
            for i, data_batch in pbar:
                # Handle batch format (retrieval vs non-retrieval models)
                if len(data_batch) == 5:
                    index, batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                else:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = data_batch
                    index = None
                
                # Move data to computation device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input (context + zero-filled prediction slots)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], 
                                   dim=1).float().to(self.device)

                # ============================================================
                # FORWARD PASS: Model-specific inference
                # ============================================================
                
                if self.args.use_retrieval:
                    # Retrieval-based models (RAFT)
                    # Standard retrieval
                    outputs = self.model(batch_x, index, mode='test')
                else:
                    # Non-retrieval models (TimeMixer, WPMixer, etc.)
                    if self.args.use_amp:
                        # Mixed precision inference
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        # Standard inference
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Extract predictions and ground truth for forecast horizon
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                
                # Move to CPU for further processing
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # ============================================================
                # INVERSE SCALING: Transform back to original scale
                # ============================================================
                
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    
                    # Handle dimension mismatch for non-retrieval models
                    # Some models may output fewer features than input
                    if not self.args.use_retrieval and outputs.shape[-1] != batch_y.shape[-1]:
                        # Tile outputs to match ground truth dimensions
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    
                    # Apply inverse transformation using fitted scaler
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                # Select appropriate feature dimension
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # Store predictions and ground truth
                preds.append(pred)
                trues.append(true)
                
                # ============================================================
                # PROCESS INPUT SEQUENCES for visualization
                # ============================================================
                
                input_seq = batch_x.detach().cpu().numpy()
                
                # Inverse transform inputs if needed
                if test_data.scale and self.args.inverse:
                    shape = input_seq.shape
                    input_seq = test_data.inverse_transform(
                        input_seq.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                inputs.append(input_seq)
                
                # ============================================================
                # COLLECT SAMPLES FOR VISUALIZATION
                # ============================================================
                
                batch_size = pred.shape[0]
                n_features = pred.shape[-1]
                
                if self.args.features in ['M', 'MS'] and n_features > 1:
                    # --- MULTIVARIATE: Visualize each feature separately ---
                    for feature_idx in range(n_features):
                        if feature_samples_collected[feature_idx] < max_samples_per_feature:
                            # Take first sample from batch for this feature
                            sample_idx = 0
                            
                            # Calculate per-sample metrics
                            sample_metrics = {
                                'mae': np.mean(np.abs(
                                    pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                                )),
                                'mse': np.mean((
                                    pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                                )**2),
                                'rmse': np.sqrt(np.mean((
                                    pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                                )**2))
                            }
                            
                            # Add to visualizer
                            visualizer.add_prediction(
                                ground_truth=true[sample_idx, :, feature_idx],
                                prediction=pred[sample_idx, :, feature_idx],
                                input_seq=(input_seq[sample_idx, :, feature_idx] 
                                          if input_seq.ndim > 2 else None),
                                sample_id=f"TimeSeries_{feature_idx+1}",
                                metrics=sample_metrics
                            )
                            feature_samples_collected[feature_idx] += 1
                else:
                    # --- UNIVARIATE: Visualize single feature ---
                    for sample_idx in range(min(batch_size, max_samples_per_feature)):
                        if feature_samples_collected[0] >= max_samples_per_feature:
                            break
                        
                        feature_idx = -1  # Last feature (target)
                        
                        # Calculate per-sample metrics
                        sample_metrics = {
                            'mae': np.mean(np.abs(
                                pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                            )),
                            'mse': np.mean((
                                pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                            )**2),
                            'rmse': np.sqrt(np.mean((
                                pred[sample_idx, :, feature_idx] - true[sample_idx, :, feature_idx]
                            )**2))
                        }
                        
                        # Add to visualizer
                        visualizer.add_prediction(
                            ground_truth=true[sample_idx, :, feature_idx],
                            prediction=pred[sample_idx, :, feature_idx],
                            input_seq=(input_seq[sample_idx, :, feature_idx] 
                                      if input_seq.ndim > 2 else input_seq[sample_idx, :]),
                            sample_id=f"Sample_{feature_samples_collected[0]+1}",
                            metrics=sample_metrics
                        )
                        feature_samples_collected[0] += 1

            pbar.close()

        # ========================================================================
        # POST-PROCESSING: Aggregate predictions and prepare for evaluation
        # ========================================================================
        
        # Concatenate all batches
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'\nTest shape: {preds.shape}')
        
        # Reshape to [n_samples, pred_len, n_features]
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # Create results directory
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Aggregate inputs for visualization
        inputs_array = np.concatenate(inputs, axis=0) if inputs else None
        
        # ========================================================================
        # DTW CALCULATION (Optional): Measure shape similarity
        # ========================================================================
        
        if self.args.use_dtw:
            print("\nCalculating DTW metrics (this may take a while)...")
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            
            for i in tqdm(range(preds.shape[0]), desc='DTW Calculation', ncols=100):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            
            dtw = np.array(dtw_list).mean()
        else:
            dtw = None
        
        # ========================================================================
        # METRIC CALCULATION: Comprehensive evaluation
        # ========================================================================
        
        # Calculate standard metrics
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        # Calculate all available metrics
        all_metrics = calculate_all_metrics(preds, trues)
        
        # Add DTW to metrics if calculated
        if dtw is not None:
            all_metrics['DTW'] = dtw
        
        # ========================================================================
        # RESULTS DISPLAY: Print comprehensive metrics
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("Test Results".center(80))
        print(f"{'='*80}")
        print(format_metrics(all_metrics))
        print(f"{'='*80}\n")
        
        # ========================================================================
        # RESULTS STORAGE: Save predictions, metrics, and visualizations
        # ========================================================================
        
        # Initialize ResultsManager for comprehensive result management
        results_manager = ResultsManager(base_results_dir='./results')
        
        # Create all standard visualizations and save results
        exp_folder = results_manager.create_all_visualizations(
            preds=preds,
            trues=trues,
            inputs=inputs_array,
            metrics=all_metrics,
            setting=setting,
            args=self.args,
            dtw_value=dtw,
            num_samples=3
        )
        
        # ========================================================================
        # INTERACTIVE VISUALIZATION: Generate HTML report
        # ========================================================================
        
        # Generate comprehensive HTML visualization using ForecastVisualizer
        html_path = os.path.join(exp_folder, 'comprehensive_results.html')
        visualizer.create_comprehensive_html(html_path, max_predictions=50)
        
        # Display visualization summary
        total_visualized = sum(feature_samples_collected)
        print(f"\n✨ Additional interactive visualization created!")
        print(f"   Total samples visualized: {total_visualized}")
        if self.args.features in ['M', 'MS']:
            print(f"   Features visualized: {n_features_to_visualize} "
                  f"(each with {max_samples_per_feature} samples)")
        print(f"   Comprehensive report: {html_path}\n")

        return