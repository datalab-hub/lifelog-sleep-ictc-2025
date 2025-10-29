# Hybrid Autoencoder-LightGBM for Sleep Health Prediction

## 📋 Overview
This repository contains the implementation of a hybrid deep learning framework that combines autoencoder-based feature extraction with LightGBM classification for predicting sleep health metrics from multimodal lifelog data.

**Publication**: Accepted at ICTC 2025 (Not published)

## 🏗️ Model Architecture

### Overall Framework
```
Raw Lifelog Data (12 Sensor Types)
    ↓
Feature Engineering & Selection (20 features)
    ↓
┌─────────────────────────────────────┐
│           Autoencoder               │
│  ┌─────────────────────────────────┐│
│  │ Encoder: 20→128→64→20 (ReLU)    ││
│  │ Decoder: 20→64→128→20 (Sigmoid) ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    ↓
Feature Fusion (20 original + 20 encoded = 40D)
    ↓
LightGBM Classifiers (6 tasks: Q1, Q2, Q3, S1, S2, S3)
```

### 1. Data Processing Pipeline

#### Sensor Modalities (12 types)
- **Mobile Sensors**: Activity, GPS, Light, WiFi, Bluetooth, Screen Usage, etc.
- **Wearable Sensors**: Heart Rate, Pedometer, Light

#### Feature Categories (20 selected features)
- Digital Behavior Patterns
- Connectivity Indicators  
- Physical Activity Levels
- Device Charging Behaviors
- Environmental Light Exposure
- Mobility Characteristics

### 2. Autoencoder Architecture
```python
# Pseudo-code structure
class SleepHealthAutoencoder:
    def __init__(self):
        # Encoder
        self.encoder = Sequential([
            Dense(128, activation='relu', input_shape=(20,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(), 
            Dropout(0.3),
            Dense(20, activation='relu')  # Latent space
        ])
        
        # Decoder
        self.decoder = Sequential([
            Dense(64, activation='relu', input_shape=(20,)),
            BatchNormalization(),
            Dropout(0.3), 
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(20, activation='sigmoid')  # Reconstruction
        ])
```

### 3. Classification Framework

#### Target Variables
- **Subjective Metrics** (Binary Classification)
  - Q1: Overall sleep quality
  - Q2: Physical fatigue 
  - Q3: Stress level

- **Objective Metrics**
  - S1: Total sleep time (3-class)
  - S2: Sleep efficiency (Binary)
  - S3: Sleep onset latency (Binary)

#### LightGBM Configuration
```python
# Hyperparameter ranges used
lgbm_params = {
    'learning_rate': [0.01, 0.03],
    'n_estimators': [500, 1000],
    'num_leaves': [50, 100], 
    'max_depth': [-1, 5],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [0, 0.01, 0.1]
}
```

## 📊 Performance Results

| Model | Public Score |
|-------|-------------|
| LSTM | 0.4158 |
| TabNet | 0.4648 |
| Random Forest | 0.5744 |
| LightGBM | 0.5822 |
| **Autoencoder + LightGBM** | **0.6069** |

## 🔧 Implementation

### Requirements
```
torch>=1.9.0
lightgbm>=3.3.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
```

### Usage Example
```python
# Load and preprocess data
data = load_lifelog_data()
features = feature_engineering(data)
selected_features = feature_selection(features, top_k=20)

# Train autoencoder
autoencoder = SleepHealthAutoencoder()
autoencoder.fit(selected_features)

# Generate enhanced features
encoded_features = autoencoder.encode(selected_features)
enhanced_features = concat([selected_features, encoded_features])

# Train LightGBM classifiers
models = {}
for target in ['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3']:
    models[target] = train_lgbm(enhanced_features, labels[target])
```

## ⚠️ Limitations
- Small dataset (10 participants)
- Missing data due to naturalistic collection
- Privacy considerations for lifelog data
- Requires multiple sensor modalities


## 📧 Contact
- Minjeong Kim: kmjng0712@gmail.com
- Seongpil Han: seukwanghsp@gmail.com
