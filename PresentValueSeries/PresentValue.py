from ifrs17.DataStructure import *
from ifrs17 import Import
from ifrs17 import Importers


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


# Workspace.Reset(x => x.ResetInitializationRules());
Workspace.InitializeFrom(DataSource)


Import.FromFile("YieldCurve.xlsx", DataSource, type_=YieldCurve)

Import.FromFile("Cashflows.xlsx", DataSource, format_=ImportFormats.Cashflow)
