# Overview

This repository provides a framework for model training and predicting with the [NY SPARCS dataset](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/q6hk-esrj/), a dataset of hospital inpatient discharges. This framework makes it easy to add new types of predictive models, as well as predict on different targets with different feature combinations. 

# Getting started
You must create the file `.config/secrets.ini`:
```
[nysparcs]
region_name = <...>
s3_bucket = <...>
socrata_app_token = <...> # optional
aws_access_key_id = <...> # optional
aws_secret_access_key = <...> # optional
```
Passing a `socrata_app_token` is only necessary if passing params to `train.py` instructing it to download the dataset via the Socrata API. Note that support for AWS credentials is still in development, so passing them is optional. 

Create the `nysparcs` conda environment and activate it:
```
conda env create --file environment.yaml
conda activate nysparcs
```
Alternatively, the conda environment can be created and other configurations can be handled by running `.config/startup.sh` in the main shell, i.e. `source .config/startup.sh`. You should review the configurations in this file and change anything to suit your preferences. At minimum, change the github identity.

You are now ready to execute the `train.py` and `predict.py` programs.

# Sample usage

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

# Docker (in progress)

*NOTE: the Docker approach is not yet functional, as the codebase needs to be modified to pass AWS credentials to the docker container.*

As an alternative to the afore-mentioned approach, the environment can also be created and executed using Docker, as follows. Note that the `.config/secrets.ini` file must still be created before building the docker image.

Build the docker image:  
```
docker build -t elicutler/nysparcs .
```  

Run the docker image:  
```
docker run --rm "$(docker image list elicutler/nysparcs -q)" <train.py or predict.py with command line args>
```

# Directions for future development
* Handle AWS permissions
* Add model monitoring
* Add tests, trigger via github actions
* Add advanced hyperparameter optimization techniques (genetic algorithms, Bayesian optimization)
