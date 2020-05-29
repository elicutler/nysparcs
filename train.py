
import logging

from train_arg_parser import TrainArgParser
from safe_dict import SafeDict
from trainer import TrainerFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  
  logger.info('Begin training...')

  parser = TrainArgParser()
  trainParams = SafeDict.fromNamespace(parser.parseArgs())
  
  logger.info(f'trainParams:\n{trainParams}')
  
  trainer = TrainerFactory.make(trainParams)
  trainer.train()
  trainer.saveModel()
  
  logger.info('Training complete')