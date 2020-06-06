
import typing as T
import logging

from safe_dict import SafeDict
from predict_arg_parser import PredictArgParser
from predictor import PredictorFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  
  from sagemaker import get_execution_role
  role = get_execution_role()
  print(f'ROLE:\n{role}')
  
  logger.info('Making predictions...')
  
  parser = PredictArgParser()
  predParams = SafeDict.fromNamespace(parser.parseArgs())
  
  logger.info(f'predParams:\n{predParams}')
  
  predictor = PredictorFactory.make(predParams)
  predictions = predictor.predict()
  
  logger.info(f'Predictions:\n{predictions}')
  

    
  