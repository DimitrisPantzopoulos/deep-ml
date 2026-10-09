import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    
    # Your code here
    y_true = np.asarray(y_true)
    
    y_sorted_scores : list[tuple[int, float]] = sorted(zip(y_true, y_scores), key=lambda x: x[1], reverse=True)

    area : float = 0

    TP      : int   = 0
    TPR     : float = 0.0
    total_p : int   = int(np.sum(y_true == 1))

    FP      : int   = 0
    FPR     : float = 0.0
    total_n : int   = int(np.sum(y_true == 0))

    for label, prob in y_sorted_scores:
        prev_tpr : float = TPR
        prev_fpr : float = FPR

        if label == 1:
            TP += 1
            TPR = TP / total_p
        else:
            FP += 1
            FPR = FP / total_n
        
        area += (1 / 2) * (prev_tpr + TPR) * (FPR - prev_fpr)
    
    return area








