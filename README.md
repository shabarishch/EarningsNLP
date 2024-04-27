# Predicting Stock Price Changes after Earnings Call
The purpose of this project is to create a machine learning model that predicts percentage change in stock prices surrounding earnings calls based on sentiment analysis of earnings call transcripts, earnings and revenue data, and stock prices and volume before the earnings call. A simple trading strategy based on the predictions was also devised.

## Team Members
- Shabarish Chenakkod
- Jasper Liang
- Shravan Patankar
- Tejaswi Tripathi

# Descriptions
## Overview
Stock prices are hard to predict. One significant factor that influences the stock market is corporate earnings announcements. They provide crucial insights into a company's financial health, often leading to movement in stock prices. We believe that a more precise understanding of earnings calls gives us better insight into the stock market. 

In this project, we implemented Linear Regression, XGBoost, and Neural Network to predict percentage changes in stock prices surrounding earnings calls. In particular, we include features we engineered based on sentiment analysis of earnings call transcripts, earnings and revenue data, and stock prices and volume before the earnings call. Based on our predictions, we want to find out factors that affect the stock price movement the day after the most, and devise a simple trading strategy.

## Data Set
- Our dataset consists of 98 diversified companies from the S&P index with a 6-year earning period. They consist of data related to
   - Earnings Call Transcripts: fetched by Python web scraping using Seeking Alpha API obtained in Rapid API. 
   - Earnings and Revenue Data: acquired also by Python web scraping using APIs found in Rapid API, including Earnings per Share (EPS), Earning Surprises, and Revenue Surprises.
   - Stock Prices and Volumes: sourced using Yahoo Finance python package, including Stock Prices 1, 7, 15 days before and after earnings, and average volume 50 days before the earnings.
- We then conducted data processing and feature engineering.
   - We organized keywords into several categories.
   - For each category, we extracted sentences (with context) and computed average sentiment scores using VADER.
   - As for other data, we look at percentage changes instead of absolute changes.
   - Our target variable is ```perc_change_next_prev``` (percentage of stock price changes next day to previous day of earnings call).
 
## Modeling Approach
Strategies implemented and results obtained are summarized below. Metrics used include Mean Squared Error (MSE) and Correlation Coefficients between the actual and predicted values.
- Baseline Model
  - Predicts no change from previous day.
  - ```MSE = 31.16 ```
- Linear Regression
  - Conducted a 0.8 - 0.2 training - test split, stratified by symbols and based on time.
  - Used ```StandardScalar``` to scale the features.
  - The model was run 1000 times, and predictions are recorded.
  - ```MSE = 24.16```, ```Correlation Coefficient = 0.18```
  - We ranked feature importance based on magnitudes of coefficient.
- XGBoost
  - Parameters are fine-tuned using ```GridSearchCV```
  - The model was run 1000 times, and predictions are recorded.
  - ```MSE = 23.85```, ```Correlation Coefficient = 0.22```
- Neural Network
  - The model was run 10 times due to computational cost.
  - ```MSE = 35.62``` (sensitive to outliers), ```Correlation Coefficient = 0.31```
 
## Simple Trading Strategy
- Positive prediction: buy and hold stock for 1 day. Negative prediction: short and then buy for 1 day.
- Obtained returns of (XGBoost 4.8%, Linear Regression 3%) by trading around 4 earnings call days. Naively buying the stock on earnings day and selling it the next gives a return of 2%.
- Note: strategy applies only to institutional investors / earnings during active hours

## Future Iterations
- Use TF-IDF vectorization on the earnings call transcripts.
- Explore more advanced machine learning models such as LSTM.
- Apply our predictive modeling techniques to other financial events and instruments.

