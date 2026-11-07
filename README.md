# Multimodal Prediction of Pet Adoption Speed

Project 2.2 (individual), Multimodal Data Analysis, Adelaide University.
Author: [your name / student ID]

We predict the AdoptionSpeed class (0-4) of a pet listing from its profile image, its text description and its tabular listing data (PetFinder.my Adoption Prediction, Kaggle, 2018). The notebook compares unimodal baselines with three fusion methods:

- **Early fusion**: frozen ResNet-50 image features + frozen multilingual BERT text features + tabular features, concatenated, one XGBoost classifier.
- **Late fusion**: one XGBoost model per modality; probabilities combined with weights chosen on the validation set.
- **Attention fusion**: a small network that treats the three modalities as tokens and mixes them with self-attention.

Evaluation uses accuracy, macro F1, weighted F1, quadratic weighted kappa (QWK) and log loss, with bootstrap confidence intervals, per-class and subgroup analysis, an overfitting check (train vs validation vs test) and a class re-weighting experiment for the rare class 0.

## Headline test-set results (from the committed notebook run)

| Model | Accuracy | Macro F1 | QWK | Log loss |
|---|---|---|---|---|
| Visual only (ResNet-50 + XGBoost) | 0.361 | 0.259 | 0.302 | 1.402 |
| Text only (mBERT + XGBoost) | 0.323 | 0.249 | 0.121 | 1.444 |
| Tabular only (XGBoost) | 0.379 | 0.289 | 0.339 | 1.353 |
| Early fusion | 0.390 | 0.298 | 0.366 | 1.355 |
| Late fusion | 0.397 | 0.296 | 0.363 | 1.354 |
| Attention fusion | 0.388 | 0.285 | 0.354 | 1.365 |

The full table with confidence intervals is in `results/test_summary.csv`. Differences between the fusion methods are within the bootstrap noise.

## Repository contents

- `petfinder_multimodal.ipynb`: the whole pipeline (data exploration, preprocessing, feature extraction, splitting, fusion models, evaluation). Run it from top to bottom.
- `requirements.txt`: package versions used.
- `results/test_summary.csv`: final test-set comparison table, written by the last notebook cell.

The repository contains **no competition data** and no data derived from it.

## How to reproduce

1. **Environment** (Python 3.13 was used; other 3.10+ versions should work):

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Data.** Create a Kaggle account, open https://www.kaggle.com/competitions/petfinder-adoption-prediction and accept the competition rules (without this the download is refused with 403). Then, with the Kaggle command-line tool configured:

   ```
   kaggle competitions download -c petfinder-adoption-prediction
   mkdir data && unzip petfinder-adoption-prediction.zip -d data
   ```

   The notebook only needs `data/train/train.csv`, `data/train_images/`, `data/breed_labels.csv` and `data/state_labels.csv`. It uses only the labelled training files; the train/validation/test split is made inside the notebook.

3. **Run.** Open `petfinder_multimodal.ipynb` and choose Restart Kernel and Run All Cells. On an Apple M4 Max (MPS) this takes about 10 minutes (feature extraction about 160 s). The first run downloads the pretrained ResNet-50 and multilingual BERT weights, so an internet connection is needed. A CPU or CUDA GPU also works but is slower. The notebook creates `features/` (cached feature arrays), `pets_split.csv` and `results/`; the first two are ignored by git.

## Reproducibility notes

- Random seeds are fixed (`SEED = 42`), and the split is made at group level so identical images or descriptions never fall in different splits.
- Exact numbers can differ slightly across hardware and library versions. XGBoost with early stopping was sensitive to tiny input changes: after masking four descriptions, some metrics moved by about 0.01 in accuracy and up to 0.05 in QWK (text-only model). Conclusions should be read with the confidence intervals in mind.
- The test set is used only for the final evaluation; models, weights and hyper-parameters were selected on the validation set.

## Data, ethics and bias

- Data: PetFinder.my Adoption Prediction (Howard, Apers and Mongrel Jedi, Kaggle, 2018), used for academic, non-commercial work under the competition rules (section 7A). It is not redistributed (section 7B).
- Privacy: descriptions are scanned for e-mail addresses and phone numbers, which are masked before text encoding. Names inside descriptions are not removed.
- Bias: class 0 is only 2.7% of the data and is never predicted by the main models; post-hoc class re-weighting improves recall for classes 0 and 3 at a cost in accuracy and QWK. The data come mainly from Selangor and Kuala Lumpur (Malaysia), so results may not generalise elsewhere. The label reflects listing outcomes, not the worth of an animal, and the model should not be used to deprioritise animals.
