# CO₂ Emissions Forecasting and Anomaly Detection API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A FastAPI-based service for forecasting future CO₂ emissions and detecting anomalies in historical data, specifically for Tunisia.

## Features

- **Historical Data**: Retrieve Tunisia's CO₂ emissions data from 1950-2022
- **Forecasting**: Predict future CO₂ emissions using an LSTM neural network
- **Anomaly Detection**: Identify unusual emission years using Isolation Forest
- **REST API**: Fully documented endpoints with OpenAPI specification
- **Scalable**: Designed for easy deployment and extension

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and model status |
| `/historical` | GET | Get historical CO₂ data for Tunisia |
| `/predict` | POST | Forecast future CO₂ emissions (1-20 years) |
| `/anomalies` | GET | Detect anomalous years in emissions data |
| `/docs` | GET | Documentation of how to use every endpoint |

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/co2-emissions-api.git
   cd co2-emissions-api

2. Install the requirements using pip
3. Start the FastAPI server
   ```bash
   uvicorn main:app --reload
The API will be available at http://localhost:8000. Visit http://localhost:8000/docs for interactive documentation.

# Models
Forecasting Model
Type: LSTM Neural Network

Input: 10 years of historical data (look_back=10)

Output: CO₂ emissions predictions for future years

Training: Trained on Tunisia's historical CO₂ data

# Anomaly Detection
Algorithm: Isolation Forest

Features: Annual CO₂ emissions

Output: Anomaly scores (-1 for anomalies, 1 for normal)

# Data Source
Processed data from Our World in Data CO₂ dataset.
  
