
import logging

logger = logging.getLogger(__name__)


TARGET_OPTIONS = ('prior_auth_dispo', 'length_of_stay')
EVAL_OPTIONS = ('roc_auc', 'pr_auc', 'mean_absolute_error', 'mean_squared_error')
FIXED_SEED = 617

S3_BUCKET = 'sagemaker-us-west-2-207070896583'
LOCAL_ARTIFACTS_PATH = 'artifacts/'