# Stock Prediction Using LSTM

## Project Overview
This project aims to predict stock prices using Long Short-Term Memory (LSTM) neural networks. The focus is on preprocessing historical stock data, calculating moving averages, and training an LSTM-based model to make accurate predictions.

## Project Structure

```
├── .vscode/                    # VSCode configuration files
├── chromedriver                # Web driver for web scraping
├── cleaned_data.csv            # Preprocessed dataset
├── dataset_generator.py        # Script to generate datasets
├── dataset_scrawling.py        # Script to scrape historical stock data
├── feature_extracter.py        # Script for feature engineering (e.g., moving averages)
├── historical_data.csv         # Raw historical stock data
├── historical_data_with_moving_averages.csv  # Dataset with added features
├── myenv/                      # Virtual environment directory
├── trainer.py                  # Model training script
└── README.md                   # Project documentation
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ye0985/Stock_Prediction.git
   cd Stock_Prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Prerequisites
- Python 3.9+
- TensorFlow
- Pandas
- Matplotlib
- Scikit-learn

### 1. Data Preprocessing
Run the data preprocessing script to clean the data and compute features:
```bash
python dataset_generator.py
python feature_extracter.py
```

### 2. Model Training
Train the LSTM model with the preprocessed dataset:
```bash
python trainer.py
```

### 3. Visualize Predictions
The `trainer.py` script will generate a plot comparing actual vs. predicted prices.

## Dataset
The dataset was created by scrawled stock data for the year 2024 from Yahoo Finance. The scrawled data was enriched with calculated 3-month and 5-month moving averages (`3MA` and `5MA`), providing additional features for training the prediction model.
```bash
df['3MA'] = df['Adj Close'].rolling(window=3).mean()
df['5MA'] = df['Adj Close'].rolling(window=5).mean()
```

### Key Features
- **All Columns:** Date, Open, High, Low, Close Close price adjusted for splits, Adj Close (Adj Close Adjusted close price adjusted for splits and dividend and/or capital gain distributions.).
- **Calculated Features:** 3MA (3-month moving average), 5MA (5-month moving average), Adj Close, Volume.
- **Target Variable:** Adj Close.

## Workflow

### 1. Data Collection
- Stock data was scrawled using `Yahoo Finance` for the year 2024 January - December.
- Calculations for 3MA and 5MA were performed to include trends in the dataset.

### 2. Data Preprocessing
- Missing or incomplete data was cleaned to ensure robust model performance.
- Normalization was applied to scale values between 0 and 1 using `MinMaxScaler`.

### 3. Model Training
- LSTM-based neural network with the following architecture:
  - LSTM Layer: 128 units with `tanh` activation.
  - Dense Layer: 1 unit with a linear activation function.
- Optimizer: Adam with a learning rate of `0.0001`.
- Loss Function: Mean Squared Error (MSE).

### 4. Evaluation
- The model's performance was evaluated using Mean Absolute Percentage Error (MAPE).
- Initial MAPE: **4.1%**, with ongoing efforts to optimize.

### 5. Visualization
Predictions vs. actual prices were visualized to analyze model performance over test data.


## Contributing

Feel free to fork the repository and submit pull requests. Contributions are welcome to enhance the accuracy and usability of the project.



