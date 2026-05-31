# Remote Sensing Land Cover Classification

**Course:** Machine Learning for Earth System Sciences (MLESS), SoSe 2026  
**Assignment:** Homework 1  
**Author:** Pelin Su Kaplan  
**University:** University of Cologne

---

## Task Description

For this homework I looked at two different ways to classify land cover from satellite images:

- **Random Forest** — a traditional ensemble method from scikit-learn
- **CNN (Convolutional Neural Network)** — a deep learning model built with PyTorch

---

## Dataset

**SAT-6** — Louisiana State University & NASA Ames Research Center  
**Source:** [B2share](https://b2share.eudat.eu/records/89654eac10724d30a6c7e51f2c5422de)

| Property | Value |
|---|---|
| Total samples | 81,000 |
| Classes | 6 |
| Image size | 28 × 28 pixels |
| Channels | 4 (R, G, B, NIR) |
| Columns per sample | 3,136 (28 × 28 × 4) |
| Label format | One-hot encoded |

**Classes:** `building`, `barren_land`, `trees`, `grassland`, `road`, `water`

**Channel layout in the flat vector:**

| Channel | Column range |
|---|---|
| Red (R) | 0 – 783 |
| Green (G) | 784 – 1,567 |
| Blue (B) | 1,568 – 2,351 |
| NIR | 2,352 – 3,135 |

> **Note:** The data files are not included in this repository. Download them using the `wget` cells provided in each notebook, or manually from the B2share link above. Place the files in a `data/` folder at the root of the repository.

---

## Repository Structure

```
├── Random_forest_classifier_on_remote_sensing_image.ipynb
├── CNN_classifier_on_remote_sensing_image.ipynb
├── README.md
├── FEEDBACK.md
├── requirements.txt
├── .gitignore
└── data/               ← not tracked by git (see .gitignore)
    ├── X_test_sat6.csv
    ├── y_test_sat6.csv
    └── sat6annotations.csv
```

---

## Task 2 — Random Forest

### 2.1 — Notebook Questions

**Q: How would you extract only the green and infrared channels?**  
Each channel occupies 784 columns (28×28). Green is at columns 784–1567, NIR at 2352–3135:
```python
X_gi = X.iloc[:, list(range(784, 1568)) + list(range(2352, 3136))]
```

**Q: What is the advantage of one-hot encoding?**  
With simple integer labels (0–5) the model might treat class 5 as "larger" than class 0, which doesn't make sense for categories like land cover types. One-hot avoids that by treating each class independently. It also works naturally with softmax outputs in neural networks.

**Q: Why `extend` here and `append` above?**  
`append` adds its argument as a single element, so you'd end up with a list of lists. `extend` unpacks the argument and adds each item individually, giving a flat list. We need a flat list to use with `iloc`.

**Q: What is wrong with the train/test split code?**  
Train and test indices are sampled independently for each class, so the same sample can appear in both. This is data leakage — if the model has already seen a test sample during training, the accuracy score is overstated.

**Q: Why shuffle the samples?**  
Without shuffling the indices are sorted by class, so the first 1000 are all buildings, the next 1000 all barren land, etc. If you then feed them into a model in that order the gradient updates are dominated by one class at a time, which makes training unstable.

### Results

**Task 2.2:** **Task 2.2 — All 4 channels (full model):** Overall accuracy **94.17%**. Per-class breakdown:
 
| Class | Precision | Recall | F1-score |
|---|---|---|---|
| building | 0.80 | 0.99 | 0.89 |
| barren_land | 0.99 | 0.96 | 0.97 |
| trees | 0.95 | 0.96 | 0.96 |
| grassland | 0.98 | 0.83 | 0.90 |
| road | 0.98 | 0.92 | 0.95 |
| water | 0.99 | 0.99 | 0.99 |
 
`water` scores highest due to its distinctive spectral signature. `grassland` and `building` have the lowest recall — they are the most commonly confused classes.

**Task 2.3 — RGB only:**  
Dropping NIR causes the accuracy to fall. The NIR channel is especially useful for vegetation because plant leaves reflect a lot of near-infrared light while roads and buildings absorb it. The model has a harder time separating trees and grassland from built surfaces without it.

**Task 2.4 — R+G+NIR:**  
Dropping Blue instead of NIR gives a smaller drop. Blue is the least discriminative of the four channels for this classification task. R+G+NIR performs noticeably better than RGB, which confirms that NIR is more valuable than Blue.

**Task 2.5 — max_depth:**  

I tested `max_depth` values of 5, 10, 20, and None (default). Deeper trees achieve higher accuracy but the gain flattens out quickly. `max_depth=10` already gets close to unconstrained performance while being more regularized. Very shallow trees (depth 5) clearly underfit. The results match my expectation that unconstrained trees would score highest in this setup, since the dataset is relatively small and overfitting isn't a major issue here.

> *The exact accuracy values and plots are shown in the executed notebook outputs.*

---

## Task 3 — CNN

### 3.1 — Architecture Overview

The CNN has three convolutional layers with 32, 64, and 128 filters. Each is followed by ReLU activation and max pooling. The output goes through a small MLP with one hidden layer (32 units) before the final 6-class output. Training uses Adam (lr=0.001) with CrossEntropyLoss for 10 epochs.

### 3.2 — Per-class Accuracy

Overall accuracy: **95.83%**. Per-class breakdown:
 
| Class | Precision | Recall | F1-score |
|---|---|---|---|
| building | 0.91 | 0.95 | 0.93 |
| barren_land | 0.99 | 0.94 | 0.96 |
| trees | 0.97 | 1.00 | 0.99 |
| grassland | 0.94 | 0.95 | 0.95 |
| road | 0.94 | 0.91 | 0.92 |
| water | 1.00 | 1.00 | 1.00 |
 
**Key difference from task 2.2:** The CNN outputs raw logits, not one-hot arrays. Class indices are obtained via `torch.argmax(logits, dim=1)`. Images must also be de-normalized before display, since the `SAT6Dataset` applies `(x - mean) / std` normalization during loading:
```python
img_display = img_tensor * std + mean   # reverse normalization
```

### 3.3 — RGB Only CNN vs. Random Forest

 
| Model / Setting | Accuracy |
|---|---|
| Random Forest — all channels | 94.17% |
| Random Forest — RGB only | 94.00% |
| Random Forest — R+G+NIR | 94.50% |
| CNN — all channels (lr=0.001) | 95.83% |
| CNN — RGB only | 94.17% |
 
The CNN outperforms the Random Forest on all channels (95.83% vs 94.17%). The CNN also loses less accuracy when NIR is removed. This makes sense because the CNN can extract spatial features like texture and shape from the remaining channels, which partially compensates for the missing spectral information. The RF only sees flat pixel values so it depends more directly on each channel.

### 3.4 — Hyperparameter: Learning Rate

**Chosen parameter:** Learning rate (`lr`) of the Adam optimizer  
**Default value:** `0.001`  
**New value:** `0.0001` (10× smaller)
 
**Justification:** The learning rate controls step size during gradient descent. A smaller LR leads to more conservative parameter updates — slower but more stable convergence, with less risk of overshooting local minima.
 
| LR | Accuracy at epoch 10 | Notes |
|---|---|---|
| 0.001 (default) | 95.83% | Fast convergence |
| 0.0001 (new) | 81.33% | More stable but not yet converged at epoch 10 |
 
With the smaller learning rate, training was more stable but slower. After 10 epochs, the model with `lr=0.0001` had not yet reached the same accuracy as the default — it achieved 81.33% vs 95.83%. Given more epochs, it would likely match or exceed the default. The validation loss curve was noticeably smoother.
 
---

## How to Run

**1. Clone the repo:**
```bash
git clone https://github.com/pelinsukk/remote-sensing-lc.git
cd remote-sensing-lc
```

**2. Install dependencies:**

If you use Anaconda:
```bash
conda install pandas numpy scikit-learn matplotlib tqdm
conda install pytorch torchvision -c pytorch
```

If you prefer pip:
```bash
pip install -r requirements.txt
```

**3. Download the data** (run the `wget` cells in each notebook, or manually):
```bash
mkdir -p data
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/X_test_sat6.csv
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/y_test_sat6.csv
wget -P data https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/sat6annotations.csv
```

```bash
# If wget is not available, use curl:
mkdir -p data
curl -L "https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/X_test_sat6.csv" -o data/X_test_sat6.csv
curl -L "https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/y_test_sat6.csv" -o data/y_test_sat6.csv
curl -L "https://b2share.eudat.eu/api/files/a697daf7-7570-44ff-854c-0fab43f2b52c/sat6annotations.csv" -o data/sat6annotations.csv
```

**4. Open the notebooks** in VSCode or Jupyter. Select the kernel that matches your environment (Anaconda base or your virtual environment).

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
torch
tqdm
ipykernel
```

---

*Author: Pelin Su Kaplan — MLESS SoSe 2026, Universität zu Köln*
