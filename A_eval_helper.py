# A_eval_helper.py  — optional, so everyone's comparison is computed the same way.
# Import these in your notebook:  from A_eval_helper import compare_distributions, lie_detector, score_on_referee
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

NUM = ['order_amount','account_age_days','hour_of_day','num_orders_last_7d',
       'delivery_distance_km','discount_applied','prev_refunds','device_change','location_mismatch']
CAT = ['payment_method']

def _prep(df, columns=None):
    X = pd.get_dummies(df[NUM+CAT], columns=CAT)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X

def compare_distributions(mine_fraud, noon_fraud, cols=('order_amount','account_age_days','num_orders_last_7d')):
    """Side-by-side histograms of your synthetic fraud vs Noon's."""
    fig, axes = plt.subplots(1, len(cols), figsize=(5*len(cols), 4))
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.hist(mine_fraud[c], bins=30, alpha=0.6, label='Mine', density=True)
        ax.hist(noon_fraud[c], bins=30, alpha=0.6, label='Noon', density=True)
        ax.set_title(c); ax.legend()
    plt.tight_layout(); plt.show()

def lie_detector(mine_fraud, noon_fraud):
    """Train a classifier to tell your fraud from Noon's. ~0.5 accuracy = indistinguishable (good)."""
    a = mine_fraud.copy(); a['_src']=0
    b = noon_fraud.copy(); b['_src']=1
    both = pd.concat([a,b], ignore_index=True)
    X = _prep(both); y = both['_src']
    from sklearn.model_selection import cross_val_score
    acc = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0), X, y, cv=3).mean()
    print(f"Lie-detector accuracy: {acc:.2f}  (0.50 = your fraud looks just like Noon's; 0.90+ = very different)")
    return acc

def score_on_referee(clf, train_columns, referee, label="model"):
    """Score a trained classifier on the real referee test set. Returns a metrics dict."""
    Xte = _prep(referee, columns=train_columns)
    pred = clf.predict(Xte)
    r = recall_score(referee['is_fraud'], pred)
    p = precision_score(referee['is_fraud'], pred, zero_division=0)
    f = f1_score(referee['is_fraud'], pred, zero_division=0)
    cm = confusion_matrix(referee['is_fraud'], pred)
    print(f"[{label}] fraud recall={r:.2f}  precision={p:.2f}  F1={f:.2f}  false positives={cm[0,1]}")
    print("  confusion [[TN FP][FN TP]]:", cm.tolist())
    return dict(recall=r, precision=p, f1=f, false_positives=int(cm[0,1]))
