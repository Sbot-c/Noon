# 📄 Data Dictionary — Noon Food Delivery Transactions
### (This is ALL you get about the real data. No sample rows, no statistics.)

Noon's fraud team cannot share real customer transactions with you. Instead, here is the **list of columns** that exist in their real transaction data, what each means, and its type. You will use **only this** to reason about what a transaction looks like and build your own data.

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `order_amount` | number (AED) | Total value of the order |
| 2 | `account_age_days` | integer | Days since this customer's account was created |
| 3 | `hour_of_day` | integer (0–23) | Hour the order was placed |
| 4 | `num_orders_last_7d` | integer | How many orders this customer placed in the last 7 days |
| 5 | `payment_method` | category | One of: `card`, `wallet`, `cod`, `applepay` |
| 6 | `delivery_distance_km` | number | Distance from the restaurant to the delivery address |
| 7 | `discount_applied` | binary (0/1) | 1 if a promo/discount code was used |
| 8 | `prev_refunds` | integer | Number of refund claims this customer has made before |
| 9 | `device_change` | binary (0/1) | 1 if the order came from a new or changed device |
| 10 | `location_mismatch` | binary (0/1) | 1 if the billing location and delivery location differ |
| 11 | **`is_fraud`** | binary (0/1) | **TARGET — 1 = the transaction was fraudulent, 0 = legitimate** |

> That's it. **It is your job to figure out** what realistic values each column takes, how the columns relate to each other, and — most importantly — **what patterns separate a fraudulent transaction from a legitimate one.** Think like an investigator.
