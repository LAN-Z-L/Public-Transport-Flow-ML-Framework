## 🚇 A MACHINE LEARNING FRAMEWORK FOR PUBLIC TRANSPORTATION INFRASTRUCTURE SYSTEM FLOW ESTIMATION, CHARACTERIZATION, AND PREDICTION

This repository presents a machine learning-based framework designed to estimate, characterize, and predict flows within public transportation infrastructure systems. Developed and validated using real-world subway data from New York City and San Francisco, the framework supports resilient and adaptive transit operations, particularly during disruptive conditions such as the COVID-19 pandemic.


## 📌 Framework Overview
The framework comprises three core components:

1. **Public Transportation OD Flow Estimation**  
   An unsupervised ML-based flow estimation method for estimating public transportation system flows, using boarding-alighting/turnstile data.
   
   📄 _Zhang, L., and K. Liu. 2024. "Unsupervised origin-destination flow estimation for analyzing COVID-19 impact on public transport mobility". *Cities*, 151, 105086._

   - A new methodology for analyzing COVID-19 impact on public transport mobility.
   - It includes a proposed unsupervised ML method to estimate public transport system OD flows from boarding-alighting data.
   - It includes a temporal-spatial analysis method to analyze OD flow changes before and during COVID-19.
   - The proposed methodology was implemented in analyzing COVID-19 impact in the New York City area.
   - The proposed methodology could support analyzing the impact of natural disasters on public transport mobility across time and space.

2. **Probabilistic Flow Characterization**  
   A Bayesian probabilistic method for characterizing public transportation system flows, i.e., modeling and inferring the purposes of public transportation flows/trips to identify the types of human needs that these trips aim to satisfy.
     
   📄 _Zhang, L., and K. Liu. “Modeling Purposes of Public Transportation Trips for Human Need-Responsive Urban Mobility Efficiency". (Under review)_
   
   - Temporal-spatial analysis of COVID-19 impact on urban mobility from the lens of public transportation systems.
   - A new Bayesian-based method is proposed to model and infer public transportation trip purposes at the hourly, station level.
   - By integrating a gravity model for space-dependent trip purpose distributions and a temporal variation model, the method enables reliable PT trip purpose inference.
   - The proposed method is implemented in the New York City area with NYC subway data, revealing temporal and spatial variations in the COVID-19 impact on subway trip purposes, with severity differing by land-use characteristics and trip purpose types.
   - The proposed method could support analyzing the impact of large-scale natural disasters on urban mobility across time and space, aiding transportation agencies in designing more resilient services.

3. **Context-Aware Long-Range Flow Prediction**
   A context-aware long-range transportation flow prediction method, which allows for predicting transportation flows – between origin and destination zones as defined by established administrative boundaries – in a long-range setting through effective and efficient spatial contextual-temporal modeling.
   
   📄 _Zhang, L., and K. Liu. “Context-Aware Long-Range Transportation Flow Prediction for Supporting Urban Mobility Informatics". (Under review)_
   
   - Jointly model spatial, contextual, and temporal dynamics for flow prediction.
   - Propose a similarity-based shared dilated convolution method for spatial modeling.
   - Introduce a new multi-context embedding method for integrated contextual modeling.
   - Use attention-based transformer for effective long-range temporal modeling.
   - Outperform state-of-the-art method in long-range transportation flow prediction.


## 📈 Impact of COVID-19 Analysis
- Proposed unsupervised machine learning-based method for public transport system origin-destination (OD) flow estimation.
  
  <img src="https://github.com/user-attachments/assets/35cad864-ec9b-454a-99b6-0d32285e33be" width="900"/>

- Impact of COVID-19 on public transport mobility:  
  Temporal-spatial changes in OD flows of New York City subway system before and during COVID-19.
  
  <img src="https://github.com/user-attachments/assets/3d6013bc-8667-4012-a094-22fab15f219c" width="700"/>

- Experiment results
  - (1) Although the overall impact of the pandemic was becoming contained as time went by, the rise of weekly new COVID-19 cases in an analysis timeframe led to an impact relatively more intensified than the previous timeframe, in terms of both impact magnitude and breadth. Yet, the intensified impact was still much more contained than the impact for the initial phase of the pandemic, when public health interventions such as “stay-at-home” orders were in place.
  - (2) The impact showed a radial pattern: the impact on the inflows to and outflows from the Manhattan neighborhood, the center of the city, became more contained when the overall impact got milder, and the impact became more intensified when the overall impact got severer.


## 📄 Publications
This framework is based on the following research:
- Zhang, L., and K. Liu. 2024. *Unsupervised origin-destination flow estimation for analyzing COVID-19 impact on public transport mobility*. *Cities*, 151, 105086.
  https://www.sciencedirect.com/science/article/abs/pii/S0264275124003007
- Zhang, L., and K. Liu. *Modeling Purposes of Public Transportation Trips for Human Need-Responsive Urban Mobility Efficiency*. *IEEE ACCESS*, under review.
- Zhang, L., and K. Liu. *Context-Aware Long-Range Transportation Flow Prediction for Supporting Urban Mobility Informatics*. *Engineering Applications of Artificial Intelligence*, under review.
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5257002
  
- Zhang, L., and Liu, K. 2022. *Semantic modeling for supporting planning decision making toward smart cities*. In *Construction Research Congress 2022*, 272–280.
- Zhang, L., and Liu, K. 2024. *Machine Learning-Based Ranking of Factors Influencing Human Movement Purposes for Supporting Human-Infrastructure Interaction Modeling*. In *Computing in Civil Engineering 2023*, 116–124.


## 🧮 Technologies Used
- Python
- PyTorch 
- Pandas / NumPy
- Scikit-learn
