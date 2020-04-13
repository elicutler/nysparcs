'''
Data source: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/q6hk-esrj
'''
import pandas as pd
import json

from sodapy import Socrata

with open('config/secrets.json', 'r') as f:
  appToken = json.load(f)['socrata']['app_token']

client = Socrata('health.data.ny.gov', appToken)

ipDischRecs = client.get('q6hk-esrj', limit=20, order=':id')

ipDischDF = pd.DataFrame.from_records(ipDischRecs)

look1 = client.get('q6hk-esrj', limit=1, order=':id', offset=0)
look2 = client.get('q6hk-esrj', limit=1, order=':id', offset=1)

def dfLoader(chunksize):
  offset = 0
  while True:
    yield client.get('q6hk-esrj', limit=chunksize, order=':id', offset=offset)
    offset += chunksize
  
loader = dfLoader(10)
df1 = pd.DataFrame.from_records(next(loader))
df2 = pd.DataFrame.from_records(next(loader))

def gen():
  i = 0
  while i < 1:
    try:
      yield i
      i += 1
    except StopIteration:
      print('no mo')
g = gen()