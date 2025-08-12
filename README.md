# 🧪 Research Portfolio – [Emma Green]

Welcome to my research portfolio!  
I am a **Postdoctoral Researcher** specializing in **data-driven modelling, and control systems.  
My work bridges **academic research** and **industrial applications**, with experience in:

- Data-driven Modelling and Control
- Secure Cloud-native Control Algorithms
- Signal Processing in Energy Harvesting Wireless Sensor Networks 
- Model Predictive Control (MPC), Linear–quadratic regulator (LQR), and Kalman Filtering (KF) Applications
- Regularized System Identification with Kernel Methods

## Contents
- 🔧 [Data-driven Modelling + Control]: In this project, we focus on an alternating minimization-based hyperparameter tuning method for SURE estimation.
  
- 🌐 [Cloud-native Secure Controller]: This project focuses on the investigation of a sequential detection policy for combating the effect of the replay attack on a cloud-native controller. The impact of such an attack is mitigated by adding random signals to the optimal control signal before the actuation process, a technique known as watermarking policy. We study the effectiveness of this joint scheme of sequential detection with watermarking in a cloud-native controller with varying levels of delay introduced in the communication network between the cloud server and the physical plant.
  
- 📈 [Signal processing with Stochastic Control]:
  
  **Subtopics:**  
  1. ⚡ *Dynamic Programming-based Energy Allocation*:  
     Developed optimal energy allocation strategies using dynamic programming to maximize throughput and prolong network lifetime under stochastic energy arrivals.  

  2. ⏱️ *Quickest Change Detection-based Delay Minimization*:  
     Designed detection policies leveraging the quickest change detection to minimize delay in event reporting while adhering to energy constraints.  

  3. 📊 *Asymptotic Analysis of Local Change Detection*:  
     Conducted asymptotic performance analysis of local change detection algorithms for distributed sensor networks under limited energy budgets.  

## 🔗 Related Research Repositories

- **Related Research Project:** [Interpretation Error in Dynamical Systems](https://github.com/cuthie/Partial-Knowledge-Observer)  
  Focuses on defining and describing a specific type of modeling error, termed the interpretation error, that may arise from the wrongful interpretation of the physical interactions within a dynamical system.

Each project includes code, results, and documentation to support reproducibility and industrial relevance.

---

## 📂 Repository Structure

```plaintext
📁 research-portfolio/
│
├── cloud-native-contoller/
│   ├── controller.py
│   ├── cloud_lqr.py
│   ├── plant.py
│   ├── k8s_delay.yml
│   ├── sigma_gamma_tilde.py
│
├── data-driven-modeling-control/
│   ├── calibrationUtilities.py
│   ├── gradientHypUtilities_v1.py
│   ├── hyperparam_tuning_v1.ipynb
│   ├── utilities.py
│
│
├── distributed-detection/
|   ├── dynamic-programming/
│   │   ├── cau_non_adap_heu_main.m
│   │   ├── fin_hrz_non_cau_dp.m
│   |── quickest_change_detection/
│   │   ├── fc_cusum.py
│   │   ├── non_causal_opt.m
│   │   ├── opt_thr_obs_pol.m
│   │   ├── thr_fun.m
│   ├── asymptotic-analysis-local-detection/
│   │   ├── norm_mult_usr_seq_harv_t_min.m
│   │   ├── norm_mult_usr_seq_harv_t_max.m
│   │   ├── norm_Seq_maj_log.m
│   │   ├── Lauricella_A.m
│

└── README.md
