
import logging

logger = logging.getLogger(__name__)


LOCAL_ARTIFACTS_PATH = 'artifacts/'
LOCAL_DATA_PATH = 'data/'
DATA_FILE = 'Hospital_Inpatient_Discharges__SPARCS_De-Identified___2009.csv'

TARGET_OPTIONS = ('prior_auth_dispo', 'length_of_stay')
EVAL_OPTIONS = ('roc_auc', 'pr_auc', 'mean_absolute_error', 'mean_squared_error')
FIXED_SEED = 617