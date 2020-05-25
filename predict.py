
import typing as T
import logging

from safe_dict import SafeDict
from predict_arg_parser import Predictor
from predictor import PredictorFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  
  logger.info('Making predictions...')
  
  parser = PredictyArgParser()
  params = SafeDict.fromNamespace(parser.parseArgs())
  
  logger.info(f'params:\n{params}')
  
  predictor = Predictor(params)
  predictions = predictor.predict()
  
  logger.info(f'Predictions:\n{predictions}')
  

    
  