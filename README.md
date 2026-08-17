# Knowledge Workers Mental Workload Prediction using Optimised ELANFIS

**Authors:** Isaac Teoh Yi Zhe, Pantea Keikhosrokiani  
**Published in:** Applied Intelligence (Springer, 2020)  
**DOI:** https://doi.org/10.1007/s10489-020-01928-5  

---

## 📌 Abstract
- Knowledge workers face high mental workload in planning and coordination.  
- Existing machine learning models predict workload but struggle with inaccuracies.  
- Deep learning and neuro-fuzzy systems offer potential improvements.  
- This study proposes **ELANFIS (Extreme Learning Adaptive Neuro-Fuzzy Inference System)** optimized with **micro-genetic algorithms (mGA)** and **particle swarm optimization (PSO)**.  
- Results show significant improvements in prediction accuracy (lower MSE and RMSE).  

---

## 🧠 Introduction
- Mental workload (MWL) impacts productivity and well-being.  
- Both underload and overload reduce performance.  
- Traditional ANFIS models suffer from dimensionality issues and high computation time.  
- ELANFIS integrates ELM concepts to reduce these limitations.  

---

## 🔍 Background
### Mental Workload Assessment
- **Subjective techniques:** NASA-TLX, SWAT, MCH.  
- **Physiological techniques:** ECG, skin conductance, facial expressions.  
- **Performance techniques:** error rate, reaction time, task completion.  

### Fuzzy Systems & Neuro-Fuzzy Models
- Fuzzy sets allow flexible membership values.  
- ANFIS combines fuzzy logic with neural networks.  
- ELANFIS improves scalability and accuracy.  

---

## ⚙️ Methodology
### Data Collection
- Dataset: **SWELL-KW** (25 participants, interns & students).  
- Sources: questionnaires (NASA-TLX), physiological sensors, computer interactions.  

### Model Design
- **ELANFIS baseline** → trained without optimization.  
- **ELANFIS + PSO** → optimized with particle swarm.  
- **ELANFIS + mGA** → optimized with micro-genetic algorithm.  
- **Proposed hybrid model** → combines PSO + mGA for enhanced optimization.  

### Evaluation Metrics
- **Mean Squared Error (MSE)**  
- **Root Mean Squared Error (RMSE)**  
- **Mean Error**  
- **Error Standard Deviation (STD)**  

---

## 📊 Results
- **ELANFIS baseline:** RMSE ≈ 3.01, MSE ≈ 9.07.  
- **ELANFIS + PSO:** improved accuracy but limited.  
- **ELANFIS + mGA:** faster convergence, better optimization.  
- **Hybrid PSO + mGA:** best performance, lowest error values.  

---

## ✅ Conclusion
- Optimized ELANFIS significantly improves prediction of mental workload.  
- Employers can use these insights to design healthier work environments.  
- Future work: explore deeper integration of metaheuristic algorithms and larger datasets.  

---

## 📂 Resources
- Dataset: [SWELL-KW](http://cs.ru.nl/~skoldijk/SWELL-KW/Dataset.html)  
- Example implementation: (https://github.com/isaacteoh25/Research)

---

## 🔑 Keywords
- Behavior Recognition  
- Optimization  
- Deep Learning  
- Neuro-Fuzzy Networks  
- Particle Swarm Optimization  
- Genetic Algorithms  
- Regression

# Research
Dissertation

Install deep learning toolbox for plotregression

Install machine learning toolbox for fcm function
