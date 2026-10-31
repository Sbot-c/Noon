# 🕵️ Intern Challenge — Fraud Detection at Noon (Food Delivery)
### Your audition for Noon's Machine Learning team

Read this whole document before you start. It tells you the story, exactly what to do, what to submit, and how you'll be judged. Work through the **phases in order** — you are **not allowed to move to Phase 2 until you have submitted Phase 1.**

---

## 1. The Story (your situation)

**Noon runs a large food-delivery app in the UAE.** Every day, thousands of orders flow through it. A **tiny fraction are fraudulent** — stolen cards, promo-code abuse, fake "my order never arrived" refund scams. These cost Noon real money.

Noon's ML team has built a **fraud-detection system**, and they want to hire an intern to help. **You are that intern candidate.** Before they let you near their real systems, they want to see if you can actually do the job.

Here's the catch — and it's the whole challenge:

> **Noon will NOT give you their real transaction data.** It's confidential customer data; interns don't get access on day one. **All they give you is the list of column names** (see `A_INTERN_data_dictionary.md`).
>
> From *just that*, they want you to **build a working fraud-detection solution** and prove you understand the problem. If you impress them, they'll reveal how *their* team solved it — and then hand you real data to test against.

There is one more thing the team mentions on the way out: *"By the way — real fraud is **rare**. Keep that in mind."* **(They won't say more. Think about what that means for how you build your data.)**

---

## 2. The Core Problem You Must Solve

Fraud is **rare**. That creates a real machine-learning problem:

- If you try to train a fraud classifier but you have **very few examples of fraud**, the model never learns what fraud looks like — it just predicts "everything is legitimate" and misses all the fraud.

**The solution Noon uses — and the technique this assignment is about — is a GAN (Generative Adversarial Network).** You will use a GAN to **generate extra, realistic synthetic fraud examples**, add them to your training data, and show that your fraud detector gets **better** because of it.

> 🎓 **Why a GAN and not "just ask an LLM to make some rows"?** A GAN actually *learns a distribution* and generates data through an adversarial Generator-vs-Discriminator process (recall the lecture). Simply asking a chatbot to "print 1000 fraud rows" is not a GAN and will not be accepted. **You must use a real GAN** (a tabular-GAN library such as CTGAN/SDV is perfect) and show its training. You *may* use an LLM to help you **write and understand your GAN code** — that's encouraged — but the GAN must be real.

---

## 3. What You Are Given vs. What You Build

**Given to you now (Phase 1):**
- `A_INTERN_data_dictionary.md` — the column list, meanings, and types. **Nothing else.**
- This brief.
- *(Optional)* `eval_helper.py` — a small helper so everyone's comparison charts/metrics are computed the same way.

**You build yourself (an LLM may help you write the code):**
- Your **seed dataset** (reasoned from the column list).
- Your **GAN** and the **synthetic fraud** it generates.
- Your **fraud classifier**.
- All your **analysis, comparisons, and conclusions.**

**Hidden for now (Noon reveals later):** Noon's own GAN data, Noon's classifier, and — last — the real data. You will NOT have these in Phase 1. That is deliberate.

---

## 4. PHASE 1 — Build Blind (do this first, then submit)

You only have the column names. Build your whole solution from reasoning.

**Step 1.1 — Reason out what a transaction looks like.**
For each column, decide realistic values and ranges. Then think hard about the key question: **what makes a transaction fraudulent?** Which columns, and which *combinations*, would signal fraud? Write your assumptions down — you'll check them later.

**Step 1.2 — Create a "seed" dataset.**
Using your assumptions, create a starter table of transactions — both **legitimate and fraudulent** — with the `is_fraud` label. This is your best guess at what Noon's real data looks like.
> ⚠️ **Decide carefully how much of your seed should be fraud.** This single choice will matter more than you think.

**Step 1.3 — Train a fraud classifier on your seed — and feel the problem.**
Train any classifier you know (e.g., decision tree, random forest, logistic regression) to predict `is_fraud`. Look at its **confusion matrix**. Note how well it catches fraud.

**Step 1.4 — Build a GAN and generate synthetic fraud.**
Use a real GAN (CTGAN/SDV recommended) to learn from your fraud examples and **generate additional synthetic fraud rows.** Keep evidence that the GAN actually trained (the training run / loss behaviour / the generator object).

**Step 1.5 — Retrain with the synthetic fraud added (your "money shot").**
Add your GAN's synthetic fraud to your training data and **retrain the classifier.** Compare **before vs. after**:
- Did it catch **more fraud**?
- What happened to **false positives** (legit orders wrongly flagged)?
- Report **recall (fraud caught), precision, and F1** — not just accuracy. **(Accuracy alone will lie to you when fraud is rare.)**

**Step 1.6 — SUBMIT Phase 1 before going further.**
Save and hand in: your synthetic data, your before/after metrics, and your written assumptions. **Only after you submit will Noon reveal Phase 2.** (No peeking — the whole point is to commit your own answer first.)

---

## 5. PHASE 2 — The Reveal: Your GAN vs. Noon's GAN

Now Noon shares **their** GAN-generated fraud data (`B_NOON_synthetic_fraud.csv`) and **their** classifier. Put your work side by side with theirs.

**Step 2.1 — Comparison EDA.** Plot the distributions of key columns (e.g. `order_amount`, `account_age_days`, `num_orders_last_7d`) for **your synthetic fraud vs. Noon's synthetic fraud**. Where do they differ?

**Step 2.2 — Check the fraud proportion.** What fraction of your data is fraud, vs. Noon's? **Is yours realistic?** (Remember the hint: real fraud is rare.)

**Step 2.3 — The "lie-detector" test.** Train a classifier to tell **your synthetic fraud apart from Noon's.** If it easily tells them apart, your fraud looks different from theirs — in which way?

**Step 2.4 — Reflect.** Write down: what did Noon's GAN capture that yours missed? What would you change?

---

## 6. PHASE 3 — The Referee: Real Data Judges Everyone

Noon now trusts you enough to share a **sample of real transactions** (`C_REFEREE_real_test.csv`) — but **only as a test set.** You never train on it; it is the neutral exam.

**Step 3.1 — Score your classifier on the real data.** Report recall, precision, F1, and the confusion matrix on the real test set.

**Step 3.2 — Score Noon's classifier on the same real data** (use `B_NOON_model.joblib` or Noon's reported benchmark).

**Step 3.3 — The final bar chart.** Put the fraud-catch results side by side: your classifier vs. Noon's, on the real data. **Who caught more real fraud? Who flagged fewer innocent orders?**

**Step 3.4 — Explain the gap.** Noon will almost certainly do better. **Why?** (Hint: whose GAN learned from real fraud, and whose learned from guesses?) This is the most important paragraph you will write.

---

## 7. PHASE 4 — Ship It: The Fraud-Check Tool

Wrap your final classifier so a Noon analyst can **type in a transaction's details and get a verdict: FRAUD or LEGIT.** A ready-made app shell (`app.py`) is provided — plug your model into it and run it locally.

---

## 8. What to Submit

1. **Your notebook** — run, with outputs visible: seed, GAN training, classifier, all comparisons.
2. **Your synthetic fraud data** (CSV).
3. **Your before/after + vs-Noon metrics** (recall/precision/F1, confusion matrices, the bar chart).
4. **A short written report (≈1 page)** answering:
   - What did you assume fraud looked like, and how right were you?
   - Did the GAN improve your fraud detection? Show the numbers.
   - How did your GAN and classifier compare to Noon's on real data — and **why**?
   - **Why couldn't you have just asked an LLM to "make more fraud rows"? What does a GAN do that that wouldn't?**
5. **Your working fraud-check tool** (`app.py` + your saved model).

---

## 9. How You'll Be Judged (Rubric — 40 marks)

| # | Criterion | Marks |
|---|---|---|
| 1 | **Reasoning the data from column names** (sensible values, thoughtful fraud assumptions) | 5 |
| 2 | **The rarity decision** (did you reason about how rare fraud should be?) | 4 |
| 3 | **A real, working GAN** (trained properly; evidence shown; not an LLM row-dump) | 7 |
| 4 | **Before/after improvement** (retrained classifier; **recall/precision/F1**, not just accuracy) | 6 |
| 5 | **Phase 2 comparison** (EDA + fraud-rate + lie-detector vs. Noon's GAN) | 5 |
| 6 | **Phase 3 on real data** (correct test-set use; honest metrics; clear bar chart) | 5 |
| 7 | **Explaining the gap** (why Noon beat you — the key insight) | 4 |
| 8 | **The fraud-check tool** (runs; gives a verdict for an input transaction) | 2 |
| 9 | **Report quality** (incl. the "why not an LLM" reflection) | 2 |

> **You are NOT graded on beating Noon.** Noon has an unfair advantage (their GAN saw real fraud). You are graded on **method, honesty, and insight** — and on how close you get.

---

## 10. Setup & Hints

**Install:** `pip install pandas numpy matplotlib seaborn scikit-learn ctgan streamlit joblib`

**Honest hints (to keep you moving):**
- When fraud is rare, **accuracy is a trap** — a model that predicts "never fraud" can be 99% accurate and useless. Always look at **recall on fraud** and **false positives.**
- Think carefully about the **mix** of fraud vs. legit in the data you create. More fraud is not automatically better.
- A GAN needs examples to learn from — generate your seed **first**, then train the GAN on it.
- Build the models first; wire up the `app.py` tool **last**.
- The LLM is your **coding copilot** — ask it to help write/debug your GAN and explain the parts you don't follow. But **you** make the decisions, and the GAN must be real.

> **Good luck. Noon is watching. 🕵️**
