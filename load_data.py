
import pandas as pd
from sodapy import Socrata

client = Socrata('health.data.ny.gov', None)

ipDischRecs = client.get('q6hk-esrj', limit=20)

ipDischDF = pd.DataFrame.from_records(ipDischRecs)
