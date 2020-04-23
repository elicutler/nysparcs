
import logging

from train_arg_parser import TrainArgParser
from trainer import TrainerFactory

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = TrainArgParser()
  args = parser.parseArgs()
  
  modelType = 'torch' if args.torch_model is not None else 'sklearn'
  modelName = args.torch_model or args.sklearn_model
  
  dataLoc = 'local' if args.local_data_path is not None else 'internet'
  dataID = args.local_data_path or args.socrata_data_key
    
  trainer = TrainerFactory.make(
    modelType, modelName, args.target, trainFromScratch,
    dataLoc, dataID, args.batch_size, args.n_workers
  )
  trainer.train(args.epochs)
  trainer.saveModel()
  
  if args.deploy:
    trainer.deployModel()
