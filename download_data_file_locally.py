
import logging
import pathlib

from sagemaker.s3 import S3Downloader
from constants import S3_BUCKET

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


def downloadDataFileLocally() -> None:
  fileName = 'Hospital_Inpatient_Discharges__SPARCS_De-Identified___2009.csv'
  s3Path = pathlib.Path(f's3://{S3_BUCKET}/{fileName}')
  localDir = 'data/'
  
  if not pathlib.os.path.exists(f'{localDir}/{fileName}'):
    S3Downloader.download(s3Path, localDir)
    logger.info(f'Downloading {fileName} from s3 to local')
  else:
    logger.warning(f'{fileName} already exists in {localDir}')
    
    
if __name__ == '__main__':
  downloadDataFileLocally()
  