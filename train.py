
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
  
  logger.info('Starting analysis...')

  parser = TrainArgParser()
  params = SafeDict.fromNamespace(parser.parseArgs(['--run_id', 'test_local']))
  
  trainer = TrainerFactory.make(params)
  trainer.train()
  
  if params.deploy:
    trainer.deployModel()

  logger.info('Analysis complete.')