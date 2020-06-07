# Overview

This repository provides a framework for model training and predicting on an Amazon EC2 instance (most conveniently via Amazon SageMaker) with the [NY SPARCS dataset](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/q6hk-esrj/), a dataset of hospital inpatient discharges. This framework makes it easy to add new types of predictive models, as well as predict on different targets with different feature combinations. 

# Preliminaries

You must have the following:
* an AWS role with permissions to read from and write to S3
* an AWS user account with permissions to assume the above AWS role
* an S3 bucket for storing model artifacts
* a [Socrata app token](https://opendata.socrata.com/profile/edit/developer_settings) (only required for downloading the NYSPARCS data via the API)

Set the following environment variables:
```
export S3_BUCKET=<S3 bucket>
export AWS_ACCESS_KEY_ID=<user account AWS access key ID>
export AWS_SECRET_ACCESS_KEY=<user account AWS secret access key>
export AWS_CONFIG_FILE=.config/aws.ini
export SOCRATA_APP_TOKEN=<your Socrata app token>
```
In `.config/aws.ini`, set `role_arn` to the role that can read from and write to S3.

# Running without Docker

Create the `nysparcs` conda environment and activate it:
```
conda env create --file environment.yaml
conda activate nysparcs
```

You are now ready to execute the `train.py` and `predict.py` programs.

# Sample usage

## Train

Train a model from a JSON config located in `run_id_store.json`:
```
python train.py --run_id test_torch_socrata_data
```
Alternatively, pass configuration parameters directly from the command line:
```
python train.py \
--target prior_auth_dispo \
--target_type binary \
--features hospital_county age_group gender race type_of_admission ccs_diagnosis ccs_procedure \
source_of_payment_1 source_of_payment_2 emergency_department_indicator total_charges \
--cat_encoder_strat one_hot \
--pytorch_model CatEmbedNet \
--socrata_data_key <...> \
--epochs 1 \
--batch_size 100 \
--train_range 100 2000 \
--val_range 5001 6000
```
Documentation on the `train.py` parameters can be obtained by running `python train.py --help`.

Generate predictions from a trained model, given a file of JSON instances. The program can select the best trained model to use given a target variable and evaluation metric:
```
python predict.py --best_model --target prior_auth_dispo --eval_metric roc_auc --instances test_instances.json
```
Alternatively, a particular trained model can be specified to use for predictions:
```
python predict.py --model_name gradient_boosting_classifier_20200529012925213000 --instances test_instances.json
```
JSON instances can alternatively be passed directly to the `--instances` command line argument.

Documentation on the `predict.py` parameters can be obtained by running `python predict.py --help`.

# Running with Docker

Build the docker image:  
```
docker build -t elicutler/nysparcs .
```  

When running with Docker, all of the aforementioned environment variables must be passed to the container, with the exception of `AWS_CONFIG_FILE` which is set during the Docker build. Passing environment variables can be accomplished with the `-e` argument, or more conveniently, by storing them in a file to be referenced with the `--env-file` argument.

In the latter case, you must create the file, e.g. `.config/env_secrets`:
```
S3_BUCKET=<S3 bucket>
AWS_ACCESS_KEY_ID=<user account AWS access key ID>
AWS_SECRET_ACCESS_KEY=<user account AWS secret access key>
SOCRATA_APP_TOKEN=<your Socrata app token>
```

Then you can run the docker image with: 
```
docker run --env-file .config/env_secrets --rm "$(docker image list elicutler/nysparcs -q)" <train.py or predict.py with command line args>
```

# Directions for future development
* Add tests, trigger via github actions
* Add model monitoring
* Add advanced hyperparameter optimization techniques (genetic algorithms, Bayesian optimization)
