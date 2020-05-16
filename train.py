
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
  
  logger.info('Starting analysis')

  parser = TrainArgParser()
  params = SafeDict.fromNamespace(parser.parseArgs())
#   params = SafeDict.fromNamespace(parser.parseArgs(['--run_id', 'torch_test_local']))
#   params = SafeDict.fromNamespace(parser.parseArgs(['--run_id', 'torch_test_cloud']))
#   params = SafeDict.fromNamespace(parser.parseArgs(['--run_id', 'sklearn_test_local']))
  
  trainer = TrainerFactory.make(params)
  trainer.train()
  trainer.saveModel()
  
  logger.info('Analysis complete')