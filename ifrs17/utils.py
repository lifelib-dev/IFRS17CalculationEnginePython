import pandas as pd
from .DataStructure import *


def get_ifrsvars(datasource):

    data = []
    for v in datasource.Query(IfrsVariable):
        data.append({
            'Partition': v.Partition,
            'DataNode': v.DataNode,
            'AocType': v.AocType,
            'Novelty': v.Novelty,
            'AmountType': v.AmountType,
            'AccidentYear': v.AccidentYear,
            'EstimateType': v.EstimateType,
            'EconomicBasis': v.EconomicBasis,
            'Value': v.Value
        })
    vars = pd.DataFrame.from_records(data)

    data = []
    for v in datasource.Query(PartitionByReportingNodeAndPeriod):
        data.append({
            'Partition': v.Id,
            'Year': v.Year,
            'Month': v.Month
        })
    part = pd.DataFrame.from_records(data, index='Partition')

    df = vars.join(part, on='Partition')
    return df.loc[:, df.columns != 'Partition']


