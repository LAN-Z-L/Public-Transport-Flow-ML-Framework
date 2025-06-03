# A MACHINE LEARNING FRAMEWORK FOR PUBLIC TRANSPORTATION INFRASTRUCTURE SYSTEM FLOW ESTIMATION, CHARACTERIZATION, AND PREDICTION

This repository presents an AI-driven machine learning framework for estimating, characterizing, and predicting public transportation system flows. Developed using real-world data from New York City and San Francisco subway systems, this framework aims to support resilient and adaptive public transit operations during both normal and disrupted conditions (e.g., pandemics).

## 🔍 Overview
The framework includes three core components:

1. **OD Flow Estimation**  
   - Uses unsupervised learning to estimate origin-destination (OD) matrices from boarding and alighting data
   - Implements a custom encoder-decoder architecture and a flow-property-based loss function

2. **Flow Characterization**  
   - Models trip purpose distribution using a hybrid probabilistic model
   - Spatial estimation via gravity model; temporal estimation via time series variation modeling

3. **Long-Range Flow Prediction**  
   - Forecasts future OD flows using a transformer-based encoder-decoder model
   - Incorporates spatial, contextual, and temporal embeddings

## 🧠 Technologies Used
- Python
- PyTorch 
- Pandas / NumPy
- Scikit-learn
- Geopandas
- Matplotlib / Seaborn
- Jupyter Notebooks

## 📁 Repository Structure
```
├── data/                    # Sample or synthetic datasets (cleaned and anonymized)
├── estimation/              # OD flow estimation models
├── characterization/        # Trip purpose inference models
├── prediction/              # Long-range forecasting models
├── utils/                   # Helper functions and shared utilities
├── notebooks/               # Exploratory and result-visualization notebooks
├── README.md                # This file
└── requirements.txt         # Environment dependencies
```

## 📊 Example Results
Include figures or plots showing:
- OD flow matrix accuracy vs ground truth
- Inferred trip purpose distribution maps
- Forecasted vs actual flows (time series)

## 📄 Publications
This framework is based on the following research:
- Zhang, L., et al. (2023). *A Machine Learning Framework for Public Transportation Infrastructure System Flow Estimation, Characterization, and Prediction*. *Cities Journal*.
- Zhang, L., & Liu, K. (2022–2024). Various conference proceedings on semantic modeling and mobility prediction.

## 📌 How to Cite
If you use this work, please cite the corresponding paper(s) listed above.

## 📬 Contact
**Lan Zhang**  
Email: lzhang29@steven.edu  

---
*This repository is maintained for educational and research dissemination purposes. For inquiries related to collaborations or applications, feel free to reach out.*
