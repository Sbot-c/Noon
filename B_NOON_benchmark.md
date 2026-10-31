# 🏢 Noon ML Team — Benchmark Results
### ⚠️ REVEAL TO INTERN ONLY AFTER THEY SUBMIT PHASE 1

This is how **Noon's own fraud team** solved the problem, so you have a real target to compare against. All numbers are measured on the **same real referee test set** you'll receive in Phase 3 (`C_REFEREE_real_test.csv`), which has a realistic, **very low fraud rate (~0.3%)**.

## The lesson in three models (all scored on the real test set)

| Model | How it was trained | Fraud **recall** (caught) | Precision | F1 | False positives |
|---|---|---|---|---|---|
| **A. Real data only** (no GAN) | Only the few real fraud examples | **0.14** | 0.90 | 0.24 | 1 |
| **B. Noon = Real + Noon's GAN fraud** | Real data **+** Noon's GAN-generated fraud | **0.82** | 0.43 | 0.56 | 71 |

**Model A is the pain:** with almost no fraud to learn from, it catches only **14%** of real fraud.
**Model B is the win:** adding Noon's GAN fraud lifts fraud caught to **82%.** That jump — **0.14 → 0.82 recall** — is the whole reason GANs matter here.

Noon's confusion matrix on the real test set (Model B):
```
                 Predicted legit   Predicted fraud
Actually legit       19864               71
Actually fraud          12               53      <- caught 53 of 65 real frauds
```

## Why Noon's GAN is strong
Noon's GAN was trained on their **real fraud examples** (rare, but real). So the synthetic fraud it produces has the **true fraud patterns and the true rarity**. A classifier trained on it generalises to real fraud.

## What to compare your work against
- **Your fraud recall vs. 0.82.** How close did you get?
- **Your false positives vs. 71.** Did your model flag far more innocent orders? (If you over-generated fraud, it will.)
- **Your synthetic fraud's proportion & patterns vs. Noon's.** Noon's is ~realistic rarity with real patterns. Is yours?

Remember: **you are not expected to beat Noon** — their GAN saw real fraud, yours saw your assumptions. The goal is to get close and to **understand the gap.**
