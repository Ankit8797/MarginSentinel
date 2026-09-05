# Model Evaluation Report

## Metrics
- **Precision:** 0.9831
- **Recall:** 0.9667
- **F1 Score:** 0.9748
- **PR-AUC:** 0.9931
- **ROC-AUC:** 0.9996

## Confusion Matrix
| | Predicted Negative (Allow) | Predicted Positive (Block) |
|---|---|---|
| **Actual Negative (Legit/Power)** | 1426 | 1 |
| **Actual Positive (Ring)** | 2 | 58 |

## Financial Cost Breakdown
Total Financial Cost on Test Set: **4600 INR**

| Segment | Cost (INR) |
|---|---|
| Legit | 4000 |
| Power User | 0 |
| Obvious Ring | 0 |
| Camouflaged Ring | 600 |

*Cost defined as FP_COST = 4000 INR, FN_COST = 300 INR*
