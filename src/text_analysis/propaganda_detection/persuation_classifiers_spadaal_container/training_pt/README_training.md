# PT training script

In order to train the PT model on SemEval2023:

* Use make_df.ipynb file to convert the SemEval data to 6 .csv files (1 file with text, 1 with spans) for each of (train, dev, test).
* Use pt_train.py to train 1 or more models on those data.

## How to use **pt_train.py:**

You must invoke the script from its directory.

* By default it will look for the following files in the folder:

  > ./st3-train_text.csv'
  >
  > ./st3-train_spans.csv'
  >
  > ./st3-dev_text.csv'
  >
  > ./st3-dev_spans.csv'
  >
  > ./st3-test_text.csv'
  >
  > .//st3-test_spans.csv'
* It will output the trained models in './models/'
* It will output the scores of the test set in the same folder.
* It will train an xlm-roberta-large model on ALL languages of the input
* The default hyperparameters (can be changed in line #38)

  > learning_rate=3e-5,
  >
  > per_device_train_batch_size=12,
  >
  > per_device_eval_batch_size=12,
  >
  > weight_decay=0.01,
  >
  > eval_steps = 150,
  >
  > max_steps=4000,
  >
  > load_best_model_at_end = True,
  >
  > metric_for_best_model = 'f1',
  >
  > save_total_limit = 5