1. `ML-EdgeIIoT-dataset.csv` = Raw dataset
2. Do feature selection using correlation (feature_selection.py + corr.py) -> `selected_features_dataset.csv`
3. Split new dataset into three parts (split.py):
   1. `global_train.csv` -> `60%`
   2. `clients.csv` -> `30%` (for clients simulation)
   3. `global_test.csv` -> `10%` (test dataset for all global models)
4. `global_train.csv` will be used to train the initial global model (with validation dataset)
5. Split global_train.csv into `train.csv` and `val.csv` for hyperparameter tuning (for example: architecture selection, epochs, ...)
6. Train final initial model (g0) with SMOTE.