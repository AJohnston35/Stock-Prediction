# Stock-Prediction

This repository contains the code for predicting stock prices for the next 5 days using an LSTM model.

## Project Overview

The project uses an LSTM (Long Short-Term Memory) model, a type of recurrent neural network, to predict the stock prices of a company for the next 5 days based on historical stock price data.

## Data

The data used for this project are historical stock prices from various companies. The data files are stored in the `csv_files` directory.

## Scripts

The `get_data` directory contains scripts for acquiring daily and intraday stock price data.

The `lstm_model.py` script contains the code for the LSTM model.

The `trader.py` script uses the predictions from the LSTM model to make trading decisions.

## Usage

To use this project, run the scripts in the following order:

1. Run the scripts in the `get_data` directory to acquire stock price data.
2. Run `lstm_model.py` to train the LSTM model and make predictions.
3. Run `trader.py` to make trading decisions based on the predictions.

## Future Work

This project is a work in progress. Future updates will include improvements to the LSTM model and the trading strategy.