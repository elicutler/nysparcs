
import typing as T
import logging

from safe_dict import SafeDict
from deploy_arg_parser import DeployArgParser
from deployer import Deployer

logging.basicConfig(
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  level=logging.INFO
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  
  logger.info('Begin deployment...')
  
  parser = DeployArgParser()
  params = SafeDict.fromNamespace(parser.parseArgs())
  
  logger.info(f'params:\n{params}')
  
  deployer = Deployer(params)
  deployer.deploy()
  
  logger.info('Deployment complete')