# Time Series Forecasting and Evaluation on City Sensing Data

A comprehensive, production-ready framework for time series forecasting with state-of-the-art deep learning models. This library implements advanced forecasting architectures including **TimeMixer**, **WPMixer** and **RAFT** with a modular, extensible design.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mahsunaltin/citysense-forecasting.git
cd citysense-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Run Your First Experiment

```bash
# Train TimeMixer on ETTh1 dataset
python run.py \
  --task_name forecast \
  --is_training 1 \
  --model TimeMixer \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --train_epochs 10
```

### Run Predefined Experiments

```bash
# Run all ETTh1 experiments with different prediction lengths
bash scripts/ETT_script/TimeMixer_ETTh1.sh

# Run pedestrian forecasting experiments
bash scripts/Pedestrian_script/RAFT.sh

# Run microclimate sensor experiments
bash scripts/Microclimate_sensors_script/RAFT.sh
```

### 📊 Analyze Results

After running experiments, generate comprehensive comparison reports:

```bash
# Compare all experiments with interactive visualizations
python scripts/compare_experiments.py

# Or specify custom directories
python scripts/compare_experiments.py --results_dir ./results --output_dir ./my_comparison
```

**What You Get:**
- 📈 Interactive Plotly visualizations (HTML)
- 📊 Performance metrics tables (Markdown & CSV)
- 🎯 Ranking visualizations for each metric
- 📝 Comprehensive experiment reports
- 🔍 Error analysis plots

---

## 📁 Project Structure

```
citysense-forecasting/
│
├── run.py                          # Main entry point for experiments
├── requirements.txt                # Python dependencies
│
├── dataset/                        # Time series datasets
│   ├── ETT-small/                 # Electricity Transformer Temperature
│   │   ├── ETTh1.csv              # Hourly data variant 1
│   │   ├── ETTh2.csv              # Hourly data variant 2
│   │   ├── ETTm1.csv              # 15-min data variant 1
│   │   └── ETTm2.csv              # 15-min data variant 2
│   ├── pedestrian/                # Melbourne pedestrian counts
│   │   └── hourly_pedestrian.csv
│   ├── microclimate-sensors/      # Environmental sensor data
│   │   └── minutes_microclimate_sensors.csv
│   └── argyle-square/             # Urban sensing data
│       └── hourly_argyle_square.csv
│
├── scripts/                        # Experiment shell scripts
│   ├── ETT_script/                # ETT dataset experiments
│   │   ├── TimeMixer_ETTh1.sh     # TimeMixer on ETTh1
│   │   ├── RAFT_ETTm1.sh          # RAFT on ETTm1
│   │   └── WPMixer_ETTh2.sh       # WPMixer on ETTh2
│   │   └── ... (16 total scripts)
│   ├── Pedestrian_script/         # Pedestrian forecasting
│   │   ├── RAFT.sh
│   │   ├── TimeMixer.sh
│   │   └── WPMixer.sh
│   ├── Microclimate_sensors_script/  # Sensor data forecasting
│   │   └── ... (4 model scripts)
│   └── Argyle_square_script/      # Urban data forecasting
│       └── ... (4 model scripts)
│
└── src/                            # Source code modules
    │
    ├── models/                     # Neural network architectures
    │   ├── timemixer.py           # TimeMixer: Multi-scale temporal mixing
    │   ├── wpmixer.py             # WPMixer: Wavelet pattern mixing
    │   └── raft.py                # RAFT: Retrieval-augmented forecasting
    │
    ├── components/                 # Reusable model components
    │   ├── attention.py           # Attention mechanisms & decomposition
    │   ├── decomposition.py       # Trend-seasonal decomposition
    │   ├── embedding.py           # Time series embeddings
    │   ├── normalization.py       # Normalization layers
    │   └── retrieval.py           # Retrieval-based memory mechanisms
    │
    ├── data/                       # Data handling
    │   ├── datasets.py            # Dataset classes (ETT, Melbourne, etc.)
    │   └── factory.py             # Dataset factory & data loaders
    │
    ├── experiments/                # Experiment management
    │   ├── base.py                # Base experiment class
    │   └── forecasting.py         # Forecasting experiment
    │
    ├── augmentation/               # Data augmentation
    │   ├── base_transforms.py     # Basic transformations
    │   ├── dtw_transforms.py      # DTW-based augmentation
    │   └── warping_transforms.py  # Time warping augmentation
    │
    ├── evaluation/                 # Model evaluation
    │   ├── metrics.py             # Standard metrics (MSE, MAE, etc.)
    │   └── dtw_metrics.py         # DTW-based metrics
    │
    ├── config/                     # Configuration management
    │   ├── arg_parser.py          # Command-line argument parser
    │   └── defaults.py            # Default hyperparameters
    │
    ├── utils/                      # Utility functions
    │   ├── logging_utils.py       # Enhanced logging system
    │   ├── formatting_utils.py    # Pretty printing & formatting
    │   ├── time_utils.py          # Time feature extraction
    │   └── training_utils.py      # Training utilities
    │
    └── visualization/              # Result visualization
        ├── plots.py               # Plotting utilities
        └── reporters.py           # Result reporting
```

---

## 🎯 Supported Models

### 1. **TimeMixer**
Multi-scale temporal mixing architecture that captures patterns at different time resolutions through hierarchical downsampling and decomposition.

**Key Features:**
- Past-decomposable mixing blocks
- Multi-scale seasonal-trend decomposition
- Hierarchical temporal feature extraction

**Best For:** General-purpose forecasting, datasets with multiple periodicities

---

### 2. **WPMixer**
Wavelet-based pattern mixer that leverages wavelet transforms to decompose time series into multiple frequency components.

**Key Features:**
- Discrete Wavelet Transform (DWT) decomposition
- Multi-resolution pattern mixing
- Token mixing for temporal dependencies

**Best For:** Data with strong frequency components, non-stationary signals

---

### 3. **RAFT** (Retrieval-Augmented Forecasting)
Retrieval-based model that leverages historical similar patterns from training data to improve predictions.

**Key Features:**
- Multi-granularity retrieval mechanism
- Period-aware pattern matching
- Dynamic fusion of retrieved patterns

**Best For:** Datasets with recurring patterns, seasonal data

---

## 📊 Datasets

### ETT (Electricity Transformer Temperature)
Standard benchmark for time series forecasting research.

- **ETTh1/ETTh2:** Hourly recorded data (7 features)
- **ETTm1/ETTm2:** 15-minute recorded data (7 features)
- **Variables:** Load, Oil Temperature (OT), and others
- **Usage:** `--data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv`

### Melbourne Pedestrian
Pedestrian count data from Melbourne city sensors.

- **Temporal Resolution:** Hourly
- **Features:** 54 sensor locations
- **Usage:** `--data melbourne --root_path ./dataset/pedestrian/ --data_path hourly_pedestrian.csv`

### Microclimate Sensors
Environmental monitoring sensor network data.

- **Temporal Resolution:** Minutes
- **Features:** Temperature, humidity, and other environmental variables
- **Usage:** `--data melbourne --root_path ./dataset/microclimate-sensors/`

### Argyle Square
Urban environmental sensing data from Argyle Square location.

- **Temporal Resolution:** Hourly
- **Features:** 8 environmental variables
- **Usage:** `--data melbourne --root_path ./dataset/argyle-square/`

---

## 🔬 Experiments & Scripts

The `scripts/` directory contains comprehensive shell scripts for reproducing all experiments:

### ETT Experiments (16 scripts)
```bash
# Example: Run all TimeMixer experiments on ETTh1
bash scripts/ETT_script/TimeMixer_ETTh1.sh

# This runs 4 experiments with pred_len: 96, 192, 336, 720
```

Each ETT script tests multiple prediction horizons (96, 192, 336, 720) automatically.

**Available combinations:**
- 4 models × 4 datasets = 16 scripts
- Each script runs 4 different prediction lengths
- **Total: 64 ETT experiments**

### Melbourne Dataset Experiments
```bash
# Pedestrian forecasting
bash scripts/Pedestrian_script/RAFT.sh

# Microclimate sensors
bash scripts/Microclimate_sensors_script/RAFT.sh

# Argyle Square
bash scripts/Argyle_square_script/TimeMixer.sh
```

**Total Melbourne experiments:** 48 (3 datasets × 4 models × 4 pred_len)

---

## 🛠️ Source Code Overview

### `src/models/` - Model Architectures
Each model file implements a PyTorch `nn.Module` with standardized interfaces:
- `__init__(self, configs)`: Initialize with configuration
- `forward(self, x_enc, ...)`: Forward pass for predictions
- Special methods for retrieval-based models (RAFT)

### `src/components/` - Reusable Components
Modular building blocks used across models:
- **Attention mechanisms:** Self-attention, cross-attention
- **Decomposition:** Seasonal-trend decomposition (moving average, DFT)
- **Embeddings:** Positional and temporal embeddings
- **Retrieval:** Historical pattern retrieval system
- **Normalization:** RevIN and other normalization techniques

### `src/data/` - Data Management
- **`datasets.py`:** Dataset classes for each data type
  - `Dataset_ETT_hour`, `Dataset_ETT_minute`
  - `Dataset_Melbourne` (pedestrian, sensors, argyle)
  - Handles train/val/test splits, scaling, time features
- **`factory.py`:** Factory functions for creating data loaders

### `src/experiments/` - Experiment Framework
- **`base.py`:** Base experiment class with common functionality
- **`forecasting.py`:** Forecasting experiment
  - Training loop with early stopping
  - Validation and testing
  - Results saving and visualization

### `src/augmentation/` - Data Augmentation
Techniques to improve model generalization:
- **Base transforms:** Jittering, scaling, rotation
- **DTW transforms:** Dynamic Time Warping-based augmentation
- **Warping:** Time axis distortion and window warping

### `src/evaluation/` - Metrics
Comprehensive evaluation metrics:
- **Standard:** MSE, MAE, RMSE, MAPE, MSPE
- **Statistical:** RSE, CORR
- **DTW-based:** DTW distance metrics

### `src/config/` - Configuration
Centralized configuration management:
- Command-line argument parsing
- Default hyperparameters
- Model-specific configuration handling

### `src/utils/` - Utilities
Helper functions for common tasks:
- **Logging:** Colored console output, file logging
- **Formatting:** Pretty printing of arguments and results
- **Time features:** Extraction of temporal features (hour, day, month, etc.)
- **Training:** Learning rate scheduling, early stopping

### `src/visualization/` - Visualization
Tools for result analysis and presentation:
- Prediction plots
- Training curves
- Metric comparisons

---

## ⚙️ Key Configuration Options

### Basic Configuration
```bash
--task_name         # Task type: forecast
--is_training       # 1 for training, 0 for testing
--model_id          # Experiment identifier
--model             # Model name: TimeMixer, WPMixer, RAFT
```

### Data Configuration
```bash
--data              # Dataset name: ETTh1, ETTh2, ETTm1, ETTm2, melbourne
--root_path         # Root directory of dataset
--data_path         # Specific data file name
--features          # Forecasting mode: M (multivariate), S (univariate), MS
--freq              # Time frequency: h (hourly), t (minutely), d (daily)
```

### Forecasting Configuration
```bash
--seq_len           # Input sequence length (default: 96)
--label_len         # Decoder start token length (default: 48)
--pred_len          # Prediction horizon (default: 96)
```

### Model Architecture
```bash
--enc_in            # Number of encoder input features
--dec_in            # Number of decoder input features
--c_out             # Number of output channels
--d_model           # Model dimension (default: 16)
--n_heads           # Number of attention heads (default: 8)
--e_layers          # Number of encoder layers (default: 2)
--d_layers          # Number of decoder layers (default: 1)
--d_ff              # Feedforward dimension (default: 32)
--dropout           # Dropout rate (default: 0.1)
```

### Training Configuration
```bash
--train_epochs      # Number of training epochs (default: 10)
--batch_size        # Batch size (default: 128)
--learning_rate     # Learning rate (default: 0.01)
--patience          # Early stopping patience (default: 10)
--num_workers       # DataLoader workers (default: 10)
```

### Model-Specific Options

**TimeMixer:**
```bash
--down_sampling_layers    # Number of downsampling layers (default: 3)
--down_sampling_window    # Downsampling window size (default: 2)
--down_sampling_method    # Method: avg, max, conv
--channel_independence    # Use independent channels (default: 0)
```

**WPMixer:**
```bash
--wavelet          # Wavelet type: db2, db4, haar
--level            # Decomposition level (default: 1)
--stride           # Stride for wavelet transform (default: 8)
```

**RAFT (Retrieval):**
```bash
--use_retrieval    # Enable retrieval mechanism (default: 1)
--n_period         # Number of periods for retrieval (default: 3)
--topm             # Top-m retrieved patterns (default: 20)
```

---

## 📈 Example Usage

### Single Experiment
```bash
python run.py \
  --task_name forecast \
  --is_training 1 \
  --model RAFT \
  --data melbourne \
  --root_path ./dataset/pedestrian/ \
  --data_path hourly_pedestrian.csv \
  --features M \
  --seq_len 168 \
  --pred_len 24 \
  --enc_in 54 \
  --train_epochs 10 \
  --batch_size 64 \
  --use_retrieval 1 \
```

### Batch Experiments
```bash
# Run all 4 models on ETTh1
for model in TimeMixer WPMixer RAFT; do
  bash scripts/ETT_script/${model}_ETTh1.sh
done
```

---

## 📊 Results

Results are automatically saved to:
```
./results/{model_name}/{dataset_name}/{experiment_setting}/
├── metrics.npy              # Evaluation metrics
├── pred.npy                 # Predictions
├── true.npy                 # Ground truth
└── checkpoint.pth           # Model checkpoint (optional)
```

Model checkpoints are saved to:
```
./checkpoints/{experiment_setting}/checkpoint.pth
```

---

## 🤝 Contributing

Contributions are welcome! This library is designed for extensibility:

### Adding a New Model
1. Create a new file in `src/models/your_model.py`
2. Implement the model class inheriting from `nn.Module`
3. Add model-specific arguments to `src/config/arg_parser.py`
4. Import in `src/models/__init__.py`

### Adding a New Dataset
1. Create dataset class in `src/data/datasets.py`
2. Update `data_provider` method in `src/data/factory.py`
3. Add dataset to `dataset/` directory
4. Create experiment scripts in `scripts/`

### Adding New Components
Add reusable components to `src/components/` for use across models.

---

## 🙏 Acknowledgments

This library builds upon cutting-edge research in time series forecasting:

- **TimeMixer:** Multi-scale temporal mixing for efficient forecasting
- **WPMixer:** Wavelet-based decomposition for frequency-domain analysis
- **RAFT:** Retrieval-augmented approaches for pattern-based prediction

Special thanks to the time series forecasting research community.

---

## 📞 Contact

For questions, issues, or collaboration:
- **GitHub Issues:** [https://github.com/mahsunaltin/citysense-forecasting](https://github.com/mahsunaltin/citysense-forecasting/issues)
- **Email:** [altinma21@itu.edu.tr](mailto:altinma21@itu.edu.tr)

---

**Hope this repo will be helpful on your forecasting tasks! 🚀📈**
