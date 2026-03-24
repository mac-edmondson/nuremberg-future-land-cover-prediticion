# Nuremberg Future Land-Cover Prediction

This project provides a tabular machine learning framework and interactive dashboard for predicting and visualizing urban land-cover composition and temporal change in Nuremberg. Leveraging multi-temporal Sentinel-2 satellite imagery and ESA WorldCover labels, the system extracts domain-informed spatial, spectral, and temporal features mapped to a 20-meter resolution grid . The pipeline evaluates various models, ultimately utilizing a high-performing Stacking Regressor ensemble to classify regions into built-up, vegetation, or water categories and quantify their structural changes over time . To bridge the gap between technical modeling and practical application, the included web interface empowers non-expert stakeholders, such as urban planners, to interactively explore historical ground-truth data, forecast future urban expansion, and assess model confidence across the city.

## Project Report

[Project Report (PDF)](docs/report/report.pdf)

### Dataset Curation and Preprocessing
The primary input consists of multi-spectral Sentinel-2 imagery (bands B1–B12)Ground truth labels are obtained from the ESA WorldCover dataset, providing globally consistent land-cover maps at high spatial resolution* **Spatial Representation**: The study area is discretized into regular grid cells of size $20\times20$ meters to balance spatial detail with computational efficiency.
* **Cloud Cover**: Issues were addressed by selecting images throughout the year with no more than 20% cloud coverage.
* **Label Management**: Due to class imbalance in Nuremberg, "Tree Cover," "Grassland," and "Cropland" were grouped into a single "Vegetation" category to improve multi-regression performance.

### Model Training and Optimization
The framework evaluates a progression of predictive models, from linear baselines to complex nonlinear ensembles.
* **Feature Engineering**: The models utilize spectral indices (NDVI, NDWI, SAVI, EVI2, MNDWI) designed to enhance specific land surface characteristics like vegetation and built-up areas. 
* **Neighborhood Context**: Local spatial dependencies are captured by augmenting each grid cell with spectral values from its neighboring cells.
* **Hyperparameter Tuning**: Optimization was performed using Optuna, a Bayesian optimization framework, to identify high-performing configurations for KNN, Random Forest, and XGBoost.
* **Stacking Ensemble**: The final system employs a Stacking Regressor using XGBoost, Random Forest, and KNN as base learners, with Linear Regression as the final estimator. This model achieved a test $R^{2}$ of 0.8882 and an F1-score of 0.9320

### Model and Post Hoc Analysis
* **Robustness Testing**: The Stacking Regressor was subjected to Gaussian noise injection at 5%, 10%, and 25% levels to simulate sensor noise. The model maintained stable generalization, retaining an $R^{2}$ of 0.8070 even under high (25%) noise conditions.
* **Temporal Horizons**: Performance is optimal at a 2-year temporal gap ($\Delta=2$), while accuracy gradually declines for larger gaps ($\Delta\ge3$) as long-term dynamics are influenced by factors outside the current feature set.
* **Interpretability (SHAP)**: Analysis using SHAP values confirmed that vegetation-related indices (EVI2, SAVI) are the most influential features. For the built-up class, the temporal "delta years" feature and neighboring spectral bands (B4 and B8) play significant roles in capturing urban expansion.

### Code and Reproducibility
The pipeline is implemented in Python using `scikit-learn` and `XGBoost`, with data processing performed via Google Earth Engine 
* **Validation**: Experiments follow a temporal hold-out strategy, training on earlier years and evaluating on later labels to reflect real-world predictive use cases.
* **Consistency**: To ensure reproducibility, the codebase uses fixed random seeds and deterministic feature engineering steps.
* **Repository**: The codebase is publicly available at [https://github.com/mac-edmondson/machine-learning-final-project-ws2025](https://github.com/mac-edmondson/machine-learning-final-project-ws2025).

### Acknowledgements
Large language models, including ChatGPT (OpenAI) and Gemini (Google), were used for assistance in writing and refining the presentation of this work We thank the European Space Agency (ESA) for Sentinel-2 and WorldCover data, and Google Earth Engine for large-scale geospatial processing

