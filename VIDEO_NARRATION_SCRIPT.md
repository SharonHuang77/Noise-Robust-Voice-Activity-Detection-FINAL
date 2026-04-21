# 🎬 Video Milestone 2 - Narration Script (Updated Structure)
## Noise-Robust Voice Activity Detection Project
**Total Runtime: 5 minutes**

## 📌 INTRODUCTION
### SPEAKER: Haidy

Hello, we are group 7.

In this Milestone 2 video, we focus on model improvements, including lazy feature extraction, Lazy MLP training, CRNN architecture, and final evaluation results.

This phase is presented as a fully collaborative pipeline.

---

## 🧠 PART 1: LAZY FEATURE EXTRACTION 
### SPEAKER: Haidy

First, lazy feature extraction.

Compared to offline feature extraction, where all features are precomputed and loaded into memory, this approach is much more memory-efficient.

In earlier work, eager loading was manageable for small experiments, but memory usage grew quickly as the dataset scaled.

*[Run cell: Step 07 lazy dataset cell in notebook demo]*

Now, features are loaded on demand instead of all at once.

For example, even with over 6.7 million frames, requesting a single sample only loads a 1331-dimensional feature vector.
A batch of 32 frames gives a tensor of shape [32, 1331], instead of loading the entire dataset.

This means we only load what we need at each step, reducing memory usage and enabling training on larger datasets.

So we keep the same MFCC features, but switch to a more scalable and memory-efficient loading strategy.

---

## 🤖 PART 2: LAZY MLP (1:25 - 2:15) — **50 seconds**

### SPEAKER: Sharon

Next is our Lazy MLP model.

This model uses the lazy data pipeline directly, allowing us to train on large noisy datasets without loading everything into memory.

*[Run cell: Single Lazy MLP training demo]*

Here, we run a one-epoch demo on a small noisy subset to verify the full pipeline. The output shows successful data loading, training on CUDA, and the model reaches about 0.78 training accuracy and 0.75 development accuracy. This run saves configuration and metrics for later comparison.

---

## 🧩 PART 3: CRNN ARCHITECTURE (2:15 - 3:15) — **60 seconds**
### SPEAKER: Sharon

Now, our CRNN model.

*[Run cell: Step 08 CRNN architecture cell]*

This model combines convolutional layers to capture local acoustic patterns and recurrent layers to model temporal context.

From the output, we see Conv1D layers followed by a bidirectional GRU, with about 311 thousand parameters, making it both expressive and efficient.

We also verify a forward pass, where an input of shape [1, 100, 121] produces frame-level predictions.

*[Run cell: CRNN training demo]*

In a one-epoch demo on a small subset, the model trains on CUDA with 30 training sequences and 10 dev sequences, reaching about 0.69 training and 0.69 dev accuracy. The run is also saved with its configuration and metrics for reproducibility.

Compared to MLP, this model captures temporal dependencies more effectively, leading to better performance on noisy data.

---

## 📊 PART 4: FINAL EVALUATION IN NOTEBOOK 09 (3:15 - 4:40) — **85 seconds**
*Switch to notebook 09 and run final comparison cells*

### SPEAKER: Jasmine

*[Open notebook 09 and run the final comparison cells]*

Our final conclusion is based on the comparison graph across four metrics: accuracy, precision, recall, and F1-score.

The key findings are:
- Models trained only on clean-domain settings drop under noisy evaluation.
- Noise-aware training improves robustness significantly.
- Lazy models keep strong performance while reducing memory load.
- Lazy CRNN gives the best overall balance on noisy conditions, especially in F1-score.
- In SNR and noise-type breakdowns, CRNN is the most stable, with lower false alarms even in harder music and low-SNR cases.

So, the final model choice is **Lazy CRNN**, supported by the graph-based comparison, with the table used as supporting numeric detail.

---
## ✅ challenge
### SPEAKER: Jasmine
One key challenge we faced was memory usage during lazy loading.

Initially, our cache was unbounded, and with multiple workers, it kept growing during training. This led to out-of-memory errors and sometimes caused training to freeze.

To debug this, we profiled memory usage and tested different numbers of workers to isolate the issue.

We fixed it by introducing a bounded LRU cache and setting the number of workers to zero for stability.

As a result, memory usage dropped from about 2–3 GB to around 100–200 MB, and training became more stable and slightly faster.

## ✅ CLOSING (4:40 - 5:00) — **20 seconds**
### SPEAKER: Jasmine

To summarize, we built a complete end-to-end pipeline for noise-robust voice activity detection.

Our results show that while the clean MLP performs well on clean data, it struggles under noisy conditions. Training on noisy data improves robustness, and the CRNN model achieves the best overall performance with strong F1 score and better handling of temporal patterns.

For future work, we plan to explore ensemble methods, evaluate on more real-world datasets, optimize models for deployment, and experiment with more advanced architectures.
