import numpy as np
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred_proba):
    # Compute True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = []
    fpr = []
    thresholds = np.linspace(0, 1, 100)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)
    plt.show()

# Example usage
y_true = np.array([0, 1, 1, 0, 1])
y_pred_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.6])
plot_roc_curve(y_true, y_pred_proba)
