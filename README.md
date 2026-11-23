# Workspace Rental Price Tier Prediction

Kaggle competition — IIT Madras BS, Machine Learning Project (Term 3, 2026)

Predict the **price tier (1–4)** of a workspace rental listing from its location, capacity, rental terms, facilities, reviews and operator details.

**Competition:** [workstation-rental-price-tier-prediction](https://www.kaggle.com/competitions/workstation-rental-price-tier-prediction)
**Kaggle notebook:** [23f3001918-notebook-2026t3](https://www.kaggle.com/code/ayushkrgupta102999/23f3001918-notebook-2026t3)

## Results

| Metric | Accuracy |
|---|---|
| Kaggle private leaderboard | **0.71590** |
| Kaggle public leaderboard | **0.72196** |
| Single LightGBM model, fold 1 (operator-grouped CV) | 0.7256 |

The private and public scores are close, which shows that the operator-grouped validation matched the real test setting and the model did not overfit.

## Dataset

- **Train:** 279,746 listings × 60 columns · **Test:** 114,538 listings
- **Target:** `price_tier`, 4 fairly balanced classes (22–29% each)
- Data is not included in this repo. Download it from the competition page.

## Key finding: the test set has new operators

Most test listings come from operators the model never sees in training:

- Only **0.6%** of test operator *identities* appear in train (a random split would give ~60%).
- **Adversarial validation** (a classifier trained to tell train from test) reached **AUC 0.73**, which confirms that train and test differ.

A normal random K-fold would leak operator information and give over-optimistic scores. So all validation uses **StratifiedGroupKFold grouped by operator identity**: every validation fold contains 100% unseen operators, which matches the real test setting.

## Approach

**1. EDA.** Target distribution, feature distributions, missing values, train/test overlap analysis.

**2. Feature engineering (400+ features)**
- Date features, rental-term duration, capacity and availability features
- Review and rating features, operator profile features
- **141 facility indicators** parsed from the facilities list (multi-label binarization)
- **44 operator-portfolio features:** statistics over all listings of the same operator. These use no labels, so they work for unseen test operators too.
- Categorical interaction features, frequency encoding, and group-relative features (a listing compared with others in its zone or class)
- **Leakage-safe target encoding** for 16 location and category keys, computed with nested, operator-grouped out-of-fold splits

**3. Models**
- **LightGBM** (multiclass, early stopping)
- **CatBoost** (native categorical handling, GPU with CPU fallback)
- Blend weights tuned on out-of-fold predictions

**4. Stage-2 stacker.** A second LightGBM model uses the stage-1 predicted probabilities of *sibling* listings (other listings from the same operator or host) as extra features.

**5. Post-processing.** Class-wise probability multipliers tuned on out-of-fold predictions to maximise accuracy.

## Tech stack

Python · Pandas · NumPy · Scikit-learn · LightGBM · CatBoost · Matplotlib

## Files

| File | Description |
|---|---|
| `notebook.ipynb` | Full pipeline: EDA → features → CV → models → stacking → submission |

## How to run

1. Download the competition data from Kaggle.
2. Open the notebook on Kaggle (GPU recommended) or locally, and set `DATA_PATH` to the data folder.
3. Set `QUICK_RUN = True` for a fast test run, or `False` for the full 5-fold training.
