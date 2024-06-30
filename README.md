# Predicting bronchopulmonary dysplasia in the neonatal intensive care
## Applying machine learning on clinically measured static data and temporal vital signs

Submitted in partial fulfillment for the degree of MSc at University of Amsterdam.

**Contact**
Daniel Szabo
daniel.szabo@student.uva.nl

**Abstract**
Medical advancements have improved the survival rates of preterm infants but have led to a rise in bronchopulmonary dysplasia (BPD). BPD is caused by disrupted lung development and prolonged respiratory support, increasing the risk of mortality and long-term health issues. Identifying newborns at high risk for BPD is crucial for administering corticosteroids to only those, preventing unnecessary treatment and risks. This research outlines the design of an innovative machine-learning model aimed at predicting BPD in preterm infants by leveraging both static clinical data and dynamic vital sign measurements from the first week post-birth. The approach involved training an LSTM autoencoder to encode four temporal features, which were then combined with static data. An ensemble model was built by connecting a fully connected network (FCN) and a temporal convolutional network (TCN) to a shared linear layer. Performance was assessed using three, five, and seven days of temporal data, with the seven-day model performing best, achieving a c-statistic of 0.83. While the results align with the findings of the state of the art, our efforts confirm the added value of integrating temporal data. With a strong focus on explainability, the design intends to ensure the model's transparency, facilitating application in clinical environment to aid decisions in neonatal care.

Please note that due to the confidential nature of patient data, running the code is only possible at the site of Amsterdam University Medical Center after prior consultation.