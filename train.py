
import logging

from argparse import ArgumentParser
from trainer import TrainerFactory
from utils import parseArgsFromRunID

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--target', type=str)
  parser.add_argument('--torch_model', type=str)
  parser.add_argument('--sklearn_model', type=str)
  parser.add_argument('--local_data_path', type=str)
  parser.add_argument('--socrata_data_key', type=str)
  parser.add_argument(
    '--train_from_scratch', action='store_true', 
    help=(
      'Do _not_ attempt to load model parameters from prior run'
      ' to initialize parameters for current run.'
    )
  )
  parser.add_argument('--epochs', type=int, default=1)
  parser.add_argument(
    '--batch_size', type=int, default=-1, 
    help='-1: Do not batch data for training. (Default: -1)'
  )
  parser.add_argument(
    '--n_workers', type=int, default=-1, 
    help='-1: Use all but one core for training. (Default: -1)'
  )
  parser.add_argument('--deploy', action='store_true')
  parser.add_argument(
    '--run_id', type=str, 
    help=(
      'Read parser arguments from json identified by run_id key.'
      ' run_id json checked for in run_config_loc.'
      ' If run_id specified, other command line arguments except'
      ' for run_config_loc are ignored.'
    )
  )
  parser.add_argument(
    '--run_config_loc', type=str, default='runConfigStore.json',
    help='Location of run_id json. (Default: runConfigStore.py)'
  )
  passedArgs = parser.parse_args()
  if passedArgs.run_id is None:
    args = passedArgs
  else:
    args = parseArgsFromRunID(passedArgs.run_id, passedArgs.run_config_loc)
  breakpoint()
  assert bool(args.torch_model) + bool(args.sklearn_model) == 1
  assert bool(args.local_data_path) + bool(args.socrata_data_key) == 1
  
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
