## Submission by

Name: Wong Shen Nan

Email: wongshennan@gmail.com

as part of the requirements for AIAP Batch 13

## Overview of submitted folder and file structure

```
-- data
  |_ survive.db
-- models
-- src
  |\* **pycache**
-- eda.ipynb
-- README.md
-- requirements.txt
-- run.sh
```

## Instructions for executing the pipeline

On the command line, in the root folder, run
`./run.sh`

You can choose to run various models instead of running all at the same time. For help, run
`python3 src/main.py -h`
to see the available options.

## Pipeline

The pipeline is as follows:

1. Extraction and loading done on the data. DB is read and data is cleaned.
2. Data is then organised and categorised into discrete variables for easier classification.
3. Run and save various ML models.

## Broad Strategy

The aim is to identify factors that contribute greatly to correctly predict the occurrence of car failure using the provided dataset for an automotive company to formulate mitigative policies. Understanding the key contributing factors can help doctors determine which trait ("features") to arrest in their prognosis.

Our data strategy is as follows:

1. Analyse data in general (understanding the dataset) - We will have a broad overview of the data provided and make intuitive assumptions about the potential causality of each feature.

2. It is likely that data imputation is required to top up any missing data and clean up any false representation. This will form the majority of our data organisation so that analysis can be undistracted.

3. In order for a clean machine learning process, we will need to encode any continuous data into numerical categories. This allows proper labelling and clear categorisation. We should ensure that the data is segregated equally to prevent clear majority bias during training.

4. Once our data is cleaned and organised, we can then derive assumptions - e.g. some features are not really helpful

5. Apply various ML models and assess their relative performance. Then we evaluate and optimize the hyperparameters.

## Key Findings

After a thorough data-exploration, we cleaned understood the relative feature importance of each feature. We also had a deeper understanding of the correlations between various datasets, as cross-examining various feature relationships gave us new insights into the dataset.
