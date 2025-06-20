## Unsupervised Origin-Destination Flow Estimation for Analyzing COVID-19 Impact on Public Transport Mobility

The outbreak of COVID-19 caused unprecedented disruptions to public transport services. As such, this paper proposes a methodology for analyzing COVID-19 impact on public transport mobility. The proposed methodology includes: (1) a new unsupervised machine learning (UML) method, which utilizes a decoder-encoder architecture and a flow property-based learning objective function, to estimate the origin-destination (OD) flows of public transport systems from boarding-alighting data; and (2) a temporal-spatial analysis method to analyze OD flow change before and during COVID-19 to unveil its impact on mobility across time and space. The validation of the UML method showed that it achieved a coefficient of determination of 0.836 when estimating OD flows using boarding-alighting data. Upon the successful validation, the proposed methodology was implemented to analyze the impact of COVID-19 on the mobility of the New York City subway system. The implementation results indicate that (1) the rise in the number of weekly new COVID-19 cases intensified the impact on the public transport mobility, but not as strongly as public health interventions; and (2) the inflows to and outflows from the center of the city were more sensitive to the impact of COVID-19.

**Keywords**: COVID-19; Urban Mobility; Origin-Destination Flow Estimation; Unsupervised Machine Learning.

## 📊 Experiment Results
Include figures or plots showing:
- Proposed methodology for analyzing COVID-19 impact on public transport mobility.
  
  <img src="https://github.com/user-attachments/assets/a820bd78-de9c-453a-a3c6-5b73bb0224ff" width="900"/>
  

- Proposed unsupervised machine learning-based method for public transport system origin-destination (OD) flow estimation.
  
  <img src="https://github.com/user-attachments/assets/35cad864-ec9b-454a-99b6-0d32285e33be" width="900"/>
  

- Weekly trends of COVID-19 new cases in the United States (Data source: CDC 2022).
  
  <img src="https://github.com/user-attachments/assets/83e31a58-5766-40b6-b8d3-264480a250dd" width="700"/>
  

- Performance results for origin-destination (OD) flow estimation model selection:  
  (a) selecting optimal number of multilayer perceptron mixer blocks;  
  (b) selecting optimal weight.
  
  <img src="https://github.com/user-attachments/assets/03bb3862-b3d8-4398-a976-c5cf093d1eeb" width="800"/>


- Performance results for proposed OD flow estimation method:  
  (a) comparison of estimated and gold-standard flows before COVID-19;  
  (b) comparison of estimated and gold-standard flows during COVID-19.
  
  <img src="https://github.com/user-attachments/assets/ca0e270e-aab3-4d14-84ca-0b4823577b27" width="800"/>
  

- Impact of COVID-19 on public transport mobility:  
  Temporal-spatial changes in OD flows of New York City subway system before and during COVID-19.
  
  <img src="https://github.com/user-attachments/assets/3d6013bc-8667-4012-a094-22fab15f219c" width="700"/>
  

- Geographical distance between origin-destination (OD) station pairs of New York City subway system.
  
  <img src="https://github.com/user-attachments/assets/57d9f011-5c95-486e-8d7e-c73c20f53c19" width="300"/>


- **The implementation results showed that:**
  - **(1)** Although the overall impact of the pandemic was becoming contained as time went by, the rise of weekly new COVID-19 cases in an analysis timeframe led to an impact relatively more intensified than the previous timeframe, in terms of both impact magnitude and breadth. Yet, the intensified impact was still much more contained than the impact for the initial phase of the pandemic, when public health interventions such as “stay-at-home” orders were in place.
  - **(2)** The impact showed a radial pattern: the impact on the inflows to and outflows from the Manhattan neighborhood, the center of the city, became more contained when the overall impact got milder, and the impact became more intensified when the overall impact got severer.

## 📁 Code Organization
```
├── models/
│ ├── mlp_mixer_BART.py # MLP-Mixer model tailored for BART data
│ └── mlp_mixer_MTA.py # MLP-Mixer model tailored for MTA data
│
├── train/
│ ├── train_with_predict_BART.py # Training and prediction script for BART data
│ └── train_with_predict_MTA.py # Training and prediction script for MTA data
```
