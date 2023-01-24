import os
import pandas as pd
from ifrs17.utils import *
from ifrs17.DataStructure import *
from ifrs17 import Import
from ifrs17 import Importers

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DataSource.Reset()

result = Import.FromFile("Dimensions.xlsx", DataSource, type_=[
    ReportingNode,
    AocType,
    DeferrableAmountType,
    AmountType,
    Scenario,
    LiabilityType,
    LineOfBusiness,
    EstimateType,
    EconomicBasis,
    Currency,
    PnlVariableType,
    BsVariableType,
    Novelty,
    Profitability,
    OciType,
    ValuationApproach,
    RiskDriver,
    ProjectionConfiguration,
    ExchangeRate
])

Import.FromFile("Dimensions.xlsx", DataSource, format_=ImportFormats.AocConfiguration)

Import.FromFile("DataNodes.xlsx", DataSource, format_=ImportFormats.DataNode)
Import.FromFile("DataNodes_CH.xlsx", DataSource, format_=ImportFormats.DataNode)
Import.FromFile("DataNodes_DE.xlsx", DataSource, format_=ImportFormats.DataNode)

# Workspace.Reset(x => x.ResetInitializationRules());
Workspace.InitializeFrom(DataSource)


Import.FromFile("YieldCurve.xlsx", DataSource, type_=YieldCurve)

Import.FromFile("Cashflows.xlsx", DataSource, format_=ImportFormats.Cashflow)
Import.FromFile("CF_CH_2021_12.xlsx", DataSource, format_=ImportFormats.Cashflow)
Import.FromFile("CF_DE_2021_12.xlsx", DataSource, format_=ImportFormats.Cashflow)
Import.FromFile("CF_DE_2022_12.xlsx", DataSource, format_=ImportFormats.Cashflow)


df = get_ifrsvars(DataSource)

# # In-Force
#
# df.loc[(df['Year'] == 2022) & (df['EconomicBasis'] == 'L') & (df['AmountType'] == 'PR') & (df['Novelty'] == 'I')]
#
# df.loc[(df['Year'] == 2022) & (df['EconomicBasis'] == 'L') & (df['AmountType'] == 'CL') & (df['Novelty'] == 'I')]
#
# # New Business
#
# df.loc[(df['Year'] == 2022) & (df['EconomicBasis'] == 'L') & (df['AmountType'] == 'PR') & (df['Novelty'] == 'N')]
#
# df.loc[(df['Year'] == 2022) & (df['EconomicBasis'] == 'L') & (df['AmountType'] == 'CL') & (df['Novelty'] == 'N')]

