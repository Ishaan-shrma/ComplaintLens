import pickle
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_auc_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

with open("notebooks/artifacts/test_72846.pkl", "rb") as f:
    df = pickle.load(f)

y_true = np.load("notebooks/artifacts/test_labels.npy")
texts = df["text"].tolist()

print(f"Total canonical test reviews: {len(y_true)}")

# 1. Linear SVM
svm = joblib.load("models/svm_pipeline.joblib")
raw_svm_preds = svm.predict(texts)
svm_preds = np.array([1 if p == "positive" or p == 1 else 0 for p in raw_svm_preds])
tn, fp, fn, tp = confusion_matrix(y_true, svm_preds).ravel()
acc = accuracy_score(y_true, svm_preds)
macro_f1 = f1_score(y_true, svm_preds, average="macro")
neg_rec = recall_score(y_true, svm_preds, pos_label=0)
neg_prec = precision_score(y_true, svm_preds, pos_label=0)
print(f"Linear SVM (tuned): Acc={acc:.4f}, Macro-F1={macro_f1:.4f}, NegRec={neg_rec:.4f}, NegPrec={neg_prec:.4f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# 2. Majority Baseline (predict class 1 / positive for all)
maj_preds = np.ones_like(y_true)
tn, fp, fn, tp = confusion_matrix(y_true, maj_preds).ravel()
acc_m = accuracy_score(y_true, maj_preds)
macro_f1_m = f1_score(y_true, maj_preds, average="macro")
print(f"Majority Baseline: Acc={acc_m:.4f}, Macro-F1={macro_f1_m:.4f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# 3. VADER Lexicon
analyzer = SentimentIntensityAnalyzer()
vader_preds = []
for t in texts:
    scores = analyzer.polarity_scores(t)
    # compound >= 0 -> positive (1), else negative (0)
    vader_preds.append(1 if scores["compound"] >= 0 else 0)

vader_preds = np.array(vader_preds)
tn, fp, fn, tp = confusion_matrix(y_true, vader_preds).ravel()
acc_v = accuracy_score(y_true, vader_preds)
macro_f1_v = f1_score(y_true, vader_preds, average="macro")
neg_rec_v = recall_score(y_true, vader_preds, pos_label=0)
neg_prec_v = precision_score(y_true, vader_preds, pos_label=0)
print(f"VADER Lexicon: Acc={acc_v:.4f}, Macro-F1={macro_f1_v:.4f}, NegRec={neg_rec_v:.4f}, NegPrec={neg_prec_v:.4f}, TN={tn}, FP={fp}, FN={fn}, TP={tp}")
