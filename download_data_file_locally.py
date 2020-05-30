
import logging
import pathlib

from sagemaker.s3 import S3Downloader
from utils import parseSecrets
from constants import LOCAL_DATA_PATH, DATA_FILE

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


def downloadDataFileLocally() -> None:
  '''
  Download the dataset from s3 to local.
  '''
  s3Bucket = parseSecrets()['s3_bucket']
  s3Path = f's3://{s3Bucket}/{LOCAL_DATA_PATH}{DATA_FILE}'
  
  if not pathlib.os.path.exists(f'{LOCAL_DATA_PATH}/{DATA_FILE}'):
    S3Downloader.download(s3Path, LOCAL_DATA_PATH)
    logger.info(f'Downloading {DATA_FILE} from s3 to local')
  else:
    logger.warning(f'{DATA_FILE} already exists in {LOCAL_DATA_PATH}')
    
    
if __name__ == '__main__':
  downloadDataFileLocally()
  