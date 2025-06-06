## Unsupervised Origin-Destination Flow Estimation for Analyzing COVID-19 Impact on Public Transport Mobility

The outbreak of COVID-19 caused unprecedented disruptions to public transport services. As such, this paper proposes a methodology for analyzing COVID-19 impact on public transport mobility. The proposed methodology includes: (1) a new unsupervised machine learning (UML) method, which utilizes a decoder-encoder architecture and a flow property-based learning objective function, to estimate the origin-destination (OD) flows of public transport systems from boarding-alighting data; and (2) a temporal-spatial analysis method to analyze OD flow change before and during COVID-19 to unveil its impact on mobility across time and space. The validation of the UML method showed that it achieved a coefficient of determination of 0.836 when estimating OD flows using boarding-alighting data. Upon the successful validation, the proposed methodology was implemented to analyze the impact of COVID-19 on the mobility of the New York City subway system. The implementation results indicate that (1) the rise in the number of weekly new COVID-19 cases intensified the impact on the public transport mobility, but not as strongly as public health interventions; and (2) the inflows to and outflows from the center of the city were more sensitive to the impact of COVID-19.

**Keywords**: COVID-19; Urban Mobility; Origin-Destination Flow Estimation; Unsupervised Machine Learning.

## 📊 Experiment Results
Include figures or plots showing:
- Proposed methodology for analyzing COVID-19 impact on public transport mobility.
  ![image](https://github.com/user-attachments/assets/a820bd78-de9c-453a-a3c6-5b73bb0224ff)
  
- Proposed unsupervised machine learning-based method for public transport system origin-destination (OD) flow estimation.
  ![image](https://github.com/user-attachments/assets/35cad864-ec9b-454a-99b6-0d32285e33be)
  
- Weekly trends of COVID-19 new cases in the United States (Data source: CDC 2022).
  ![image](https://github.com/user-attachments/assets/83e31a58-5766-40b6-b8d3-264480a250dd)
  
- Performance results for origin-destination (OD) flow estimation model selection: (a) selecting optimal number of multilayer perceptron mixer blocks; (b) selecting optimal weight.
  ![image](https://github.com/user-attachments/assets/5ade136e-15f5-4d00-8552-c76cf388f1bb)
  
- Performance results for proposed OD flow estimation method: (a) comparison of estimated and gold-standard flows before COVID-19; (b) comparison of estimated and gold-standard flows during COVID-19.
  ![image](https://github.com/user-attachments/assets/ca0e270e-aab3-4d14-84ca-0b4823577b27)
  
- Impact of COVID-19 on public transport mobility: Temporal-spatial changes in origin-destination (OD) flows of New York City subway system before and during COVID-19.
  ![image](https://github.com/user-attachments/assets/3d6013bc-8667-4012-a094-22fab15f219c)
  
- Geographical distance between origin-destination (OD) station pairs of New York City subway system.
  ![image](https://github.com/user-attachments/assets/1bc8bff7-4542-42c7-a98b-e5ac601d3cf1)
