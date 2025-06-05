# Game Completion Predictor

This project aims to analyze and predict whether a video game is likely to be *completed* or *dropped* by players, using publicly available game metadata and behavioral metrics.

## Overview

The dataset is collected using the RAWG API and enhanced with estimated completion time from [HowLongToBeat](https://howlongtobeat.com/). After preprocessing and feature engineering (e.g., tag encoding, ESRB rating processing), we applied exploratory data analysis (EDA), unsupervised topic modeling with BERTopic, and wordclouds to understand player behavior and game characteristics.

Based on this enriched dataset, we created a binary classification model (`complete` vs. `dropped`). The label is defined using a heuristic: a game is considered *completed* if the average player playtime reaches at least 80% of the estimated completion time.

## Key Features

- **Data Crawling:** Uses the RAWG API to collect data on 1,000+ games, enriched with playtime and community status statistics.
- **EDA & Topic Modeling:** Visual exploration of tag distributions, completion/drop trends, and common gameplay themes via BERTopic.
- **Labeling Strategy:** Smart estimation of game completion status based on playtime vs. howlongtobeat estimates.
- **Model Training:** Applies SMOTE to address class imbalance and uses machine learning pipelines (including feature encoding, scaling, train/test split) to predict game outcomes.

## Folder Structure

- `rawg_full.ipynb`: The main notebook that contains the full analysis, model training, and insights.
- `test_api.ipynb`: A separate notebook for testing and crawling data from the RAWG API.

## Requirements

- `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
- `bertopic`, `wordcloud`, `imbalanced-learn`, `requests`

## Future Work

- Add real-time prediction API
- Improve labeling with user-level data
- Experiment with LLM-based description embeddings

