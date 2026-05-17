# Arabic-Medical-QA — Research Documentation
## Project Overview:
**Task:** Arabic Medical Question Classification  
**Goal:** Reproduce the methodology of the paper *"AraBERTv2 for Arabic Question Answering"* (OSACT 2024) on a new Arabic Medical Q&A dataset, and compare performance across three models.  
**Dataset:** Arabic Medical Q&A Dataset — Kaggle  
**Pipeline:** Data Exploration → Data Cleaning → Modeling → Comparison

## 1. Dataset Description
[Arabic Medical Q&A Dataset on Kaggle](https://www.kaggle.com/datasets/yassinabdulmahdi/arabic-medical-q-and-a-dataset)
| Split | Raw Samples | After Cleaning |
|-------|-------------|----------------|
| Train | 52,758 | 24,499 |
| Validation | 17,586 | 10,596 |
| Test | 17,586 | 10,505 |

**Columns:** `question`, `answer`, `label` (medical specialty)  
**Task Type:** Multi-class Classification (8 medical specialties after cleaning)

### Label Distribution (After Cleaning)
The dataset contains 8 medical specialties:
- ارتفاع_ضغط_الدم
- الاورام_الخبيثة_والحميدة
- امراض_الجهاز_التنفسي
- امراض_الدم
- امراض_الغدد_الصماء
- جراحة_العظام
- جراحة_عامة
- مرض_السكر

---

## 2. Data Exploration (EDA) — Issues Found

During exploratory data analysis, the following issues were identified:

| Issue | Details |
|-------|---------|
| **Null Values** | Train: 101 nulls in `answer` column |
| **Duplicate Rows** | ~40% duplicates across all splits |
| **Inconsistent Labels** | Same specialty written differently (e.g. `السكري` vs `مرض_السكري`, dashes vs underscores) |
| **Mislabeled Samples** | Samples whose content does not match their assigned label |
| **Class Imbalance** | Imbalance ratio of 1192x between largest and smallest class |

---

## 3. Data Cleaning Pipeline

### 3.1 Null & Duplicate Removal
- Dropped rows with null values in `question`, `answer`, or `label`
- Removed duplicate rows based on `question` + `answer` pairs
- Result: Train reduced from 52,758 → retained samples only

### 3.2 Label Normalization
Inconsistent label names were unified using the following rules:
- Normalized Arabic Alef variants (إ أ آ → ا)
- Replaced dashes, spaces, and underscores with a single underscore
- Applied manual mapping for semantically identical labels:

```python
label_mapping = {
    'السكري':         'مرض_السكري',
    'الدم':           'امراض_الدم',
    'الغدد_الصماء':   'امراض_الغدد_الصماء',
    'الجهاز_التنفسي': 'امراض_الجهاز_التنفسي',
}
```

- Result: Labels reduced from 37 → 8 unique specialties

### 3.3 Mislabeled Sample Detection (Cleanlab)
We used **Cleanlab** (Northcutt et al., 2021) to automatically detect mislabeled samples:
- Trained a Logistic Regression model on TF-IDF features
- Cleanlab identified samples where the model's confident prediction disagreed with the assigned label
- Suspicious samples were removed from the training set

> Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident Learning: Estimating Uncertainty in Dataset Labels. *Journal of Artificial Intelligence Research*, 70, 1373–1411.

### 3.4 Class Imbalance — Class Weights Strategy
Rather than removing data (undersampling) or duplicating samples (oversampling), we used **Class Weights** inside the model:
- Computed balanced class weights using `sklearn.utils.class_weight.compute_class_weight`
- Applied weights via a custom `WeightedTrainer` that overrides the loss function
- This penalizes the model more when it misclassifies minority classes

```python
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

**Why Class Weights over other methods?**
- Undersampling would lose 80%+ of training data
- Oversampling with only 8 samples in smallest class would cause severe overfitting
- Class Weights uses all available data while balancing model attention

---

## 4. Models

### 4.1 Baseline — TF-IDF + Logistic Regression
- **Purpose:** Establish a non-neural baseline
- **Features:** Character-level TF-IDF (1-2 ngrams, 50K features)
- **Input:** question + "[SEP]" + answer
- **Training time:** Seconds

### 4.2 AraBERTv2 (Reproduction)
- **Model:** `aubmindlab/bert-base-arabertv2`
- **Purpose:** Reproduce the original paper's methodology on new data
- **Preprocessing:** ArabertPreprocessor + Arabic text cleaning
- **Input:** Tokenized question + answer pairs (max 256 tokens)
- **Training:** 3 epochs, learning rate 2e-5, batch size 16, fp16
- **Best checkpoint:** Epoch 1 (selected by `load_best_model_at_end=True`)

### 4.3 CAMeL-BERT (LLM Comparison)
- **Model:** `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- **Purpose:** Compare against a more recent Arabic BERT model
- **Training:** Same hyperparameters as AraBERTv2
- **Best checkpoint:** Epoch 1 (selected automatically)

**Both BERT models are Encoder-only architectures** fine-tuned for sequence classification.

---

## 5. Results

| Model | Accuracy | F1-Weighted | F1-Macro |
|-------|----------|-------------|----------|
| Baseline (TF-IDF + LR) | 0.727 | 0.727 | 0.716 |
| AraBERTv2 (Reproduction) | 0.751 | 0.751 | 0.746 |
| CAMeL-BERT (LLM) | **0.752** | **0.752** | **0.747** |

### Key Observations

**1. Impact of Data Cleaning:**
Before cleaning, AraBERTv2 achieved F1-Macro of only 17%, primarily due to the 1192x class imbalance and label inconsistencies. After cleaning and applying Class Weights, F1-Macro improved to 74.6%, demonstrating the critical importance of data quality.

**2. Baseline vs BERT:**
The TF-IDF baseline achieved performance close to BERT models (71.6% vs 74-75% F1-Macro). This suggests the task is partially keyword-driven — medical specialty keywords in questions and answers provide strong classification signals even without contextual understanding.

**3. AraBERTv2 vs CAMeL-BERT:**
CAMeL-BERT marginally outperformed AraBERTv2 across all metrics (~0.1% difference), consistent with its training on more diverse Arabic corpora. However, the difference is not statistically significant.

**4. Overfitting:**
Both BERT models showed a pattern where Training Loss decreased rapidly while Validation Loss increased after Epoch 1. This is attributed to the relatively small training set (24,499 samples) after cleaning. The best model checkpoint from Epoch 1 was automatically selected using `load_best_model_at_end=True`.

---

## 6. Limitations

1. **Small training set after cleaning:** Removing duplicates (40%) and mislabeled samples significantly reduced training data, potentially limiting model capacity.

2. **Overfitting after Epoch 1:** BERT models overfit quickly on the cleaned dataset. Future work could address this with data augmentation or larger pre-training corpora.

3. **Dataset quality:** The original Kaggle dataset contained significant noise (40% duplicates, inconsistent labels), which may have affected results even after cleaning.

4. **Class imbalance remains:** Even with Class Weights, some minority classes (e.g., امراض_الدم with 431 test samples) show lower F1 scores (0.70) compared to majority classes.

---

## 7. Conclusion

This work successfully reproduced the AraBERTv2 methodology from the OSACT 2024 paper on a new Arabic Medical Q&A dataset. The key contributions are:

1. A thorough data cleaning pipeline addressing label inconsistencies, duplicates, and mislabeled samples using Cleanlab
2. A class imbalance solution using Class Weights that improved F1-Macro from 17% to 74%+
3. A three-way comparison showing that while BERT models outperform the baseline, the margin is small, suggesting the task benefits significantly from lexical features alone

**Future Work:**

The findings of this study open several research directions worthy of investigation in future work:

**1. Cross-domain Generalization**  
This study focused on a fixed set of medical specialties. Future research should investigate whether models trained on general medical questions can generalize to rare or underrepresented specialties (e.g., nuclear medicine, genetic disorders), particularly under low-resource conditions.

**2. Low-resource Medical Specialties (Few-shot Learning)**  
The significant performance gap observed in minority classes raises the question of whether few-shot or zero-shot learning approaches (e.g., prompt-based fine-tuning) can improve classification of rare medical specialties without requiring large labeled datasets.

**3. Multilingual and Code-switching Medical QA**  
Medical practitioners in Arabic-speaking countries frequently mix Arabic and English terminology. Future work should examine model robustness under code-switching conditions, where questions contain a blend of Arabic and English medical terms.

**4. Model Explainability in Clinical Decision Support**  
Given the sensitivity of medical applications, future research should address the interpretability of classification decisions using explainability frameworks such as LIME or SHAP. Understanding why a model assigns a question to a particular specialty is critical for clinical trust and deployment.

**5. Bias Analysis in Arabic Medical Language Models**  
The class imbalance observed in this dataset may reflect real-world disparities in healthcare question distribution. Future work should systematically analyze whether pre-trained Arabic language models inherit or amplify such biases, and propose debiasing strategies appropriate for the medical domain.

---
## 8. References

1. Antoun, W., Baly, F., & Hajj, H. (2020). [AraBERT: Transformer-based Model for Arabic Language Understanding](https://aclanthology.org/2020.lrec-1.299/). *LREC 2020*.
2. Inoue, G., et al. (2021). [CAMeL Tools: An Open Source Python Toolkit for Arabic Natural Language Processing](https://aclanthology.org/2021.eacl-demos.35/). *EACL 2021*.
3. Northcutt, C., Jiang, L., & Chuang, I. (2021). [Confident Learning: Estimating Uncertainty in Dataset Labels](https://jair.org/index.php/jair/article/view/12118). *JAIR*, 70, 1373–1411.
4. Original Paper: [OSACT 2024 — AraBERTv2 for Arabic QA](https://github.com/asafaya/bert-arabic-qa) *(Note: You can replace this link with the specific paper URL if available)*.
