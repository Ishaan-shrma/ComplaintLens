# Sample Reviews Dataset (`sample_reviews_5k.csv`)

## Overview
This directory contains a lightweight, anonymized 5,000-review subset of the Amazon Fine Food Reviews corpus designed for quick demonstration, local vector search, and CI/testing without requiring the full 484 MB dataset (`reviews_cleaned.csv`).

## Specifications
- **Row count**: 5,000 reviews
- **File size**: 1.70 MB (< 2.0 MB)
- **Random seed**: 42
- **Sampling method**: Stratified random sampling matching the full dataset sentiment distribution.

## Class Mix & Distribution
The sample preserves the natural class proportions of the deduplicated Amazon Fine Food corpus:

| Sentiment Class | Star Rating | Sample Count ($n=5,000$) | Percentage | Full Corpus Share ($N=393,890$) |
| :--- | :---: | :---: | :---: | :---: |
| **Positive** | 4 – 5 stars | 3,897 | 77.94% | 77.95% |
| **Negative** | 1 – 2 stars | 725 | 14.50% | 14.50% |
| **Neutral** | 3 stars | 378 | 7.56% | 7.56% |
| **Total** | | **5,000** | **100.0%** | **100.0%** |

## Schema & Privacy Protections
In accordance with read-only prototype security guidelines, all personal identifiers (`UserId`, `ProfileName`) have been dropped:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Id` | Integer | Original review ID from the Amazon Fine Food Reviews dataset. |
| `Score` | Integer | Customer star rating (1 to 5). |
| `Time` | Integer | POSIX timestamp of review submission. |
| `Summary` | Text | Short review title/headline. |
| `Text` | Text | Full review body text. |

## Generation Script
Generated using stratified sampling over clean reviews ($\le 750$ characters, retaining 86.5% of reviews) to maintain realistic complete text without truncation while guaranteeing file size $< 2.0$ MB.
