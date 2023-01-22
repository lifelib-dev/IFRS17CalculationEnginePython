from ifrs17.DataStructure import *
from ifrs17 import Import
from ifrs17 import Importers

#!import "DataModel/DataStructure"
#!import "Report/ReportMutableScopes"
#!import "Import/Importers"
#!import "Export/ExportConfiguration"
#!import "Utils/TestHelper"


#!eval-notebook "InitSystemorphRefDataToMemory"

"""
await DataSource.SetAsync();
DataSource.Reset(x => x.ResetCurrentPartitions());

Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());
Workspace.InitializeFrom(DataSource);
"""

"""
await Import.FromFile("../Files/Dimensions.csv")
                .WithType<Novelty>()
                .WithType<AocType>()
                .WithType<PnlVariableType>()
                .WithType<BsVariableType>()
                .WithType<AmountType>()
                .WithType<DeferrableAmountType>()
                .WithType<RiskDriver>()
                .WithType<EconomicBasis>()
                .WithType<EstimateType>()
                .WithType<ValuationApproach>()
                .WithType<LineOfBusiness>()
                .WithType<OciType>()
                .WithType<LiabilityType>()
                .WithType<Profitability>()
                .WithType<Currency>()
                .WithType<Partner>()
                .WithType<CreditRiskRating>()
                .WithType<Scenario>()
                .WithType<ProjectionConfiguration>()
                .WithTarget(DataSource)
                .ExecuteAsync()
"""

result = Import.FromFile("Files/Dimensions.xlsx", DataSource, type_=[
    ReportingNode,
    Novelty,
    AocType,
    PnlVariableType,
    BsVariableType,
    AmountType,
    DeferrableAmountType,
    RiskDriver,
    EconomicBasis,
    EstimateType,
    ValuationApproach,
    LineOfBusiness,
    OciType,
    LiabilityType,
    Profitability,
    Currency,
    # Partner,
    # CreditRiskRating,
    Scenario,
    ProjectionConfiguration,
    ExchangeRate
])


Import.FromFile("Files/Partner&CreditRating.xlsx", DataSource, type_=[
    Partner,
    CreditRiskRating
])


Import.FromFile("Files/Dimensions.xlsx", DataSource, format_=ImportFormats.AocConfiguration)


# Import Parameters

Import.FromFile("Files/YieldCurve.xlsx", DataSource, type_=YieldCurve)


Import.FromFile("Files/PartnerRating.xlsx", DataSource, type_=PartnerRating)    #.WithType<>().WithTarget().ExecuteAsync()
Import.FromFile("Files/CreditDefaultRate.xlsx", DataSource, type_=CreditDefaultRate)    #.WithType<>().WithTarget().ExecuteAsync()


# Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());

#!eval-notebook "InitSystemorphBaseToMemory"

# Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());
Workspace.InitializeFrom(DataSource)


Import.FromFile("Files/DataNodes_CH.xlsx", DataSource, format_=ImportFormats.DataNode)
Import.FromFile("Files/DataNodeStates_CH_2020_12.xlsx", DataSource, format_=ImportFormats.DataNodeState)
Import.FromFile("Files/DataNodeParameters_CH_2020_12.xlsx", DataSource, format_=ImportFormats.DataNodeParameter)

# Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());

#!eval-notebook "../Initialization/InitSystemorphToMemory"

# Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());
# Workspace.InitializeFrom(DataSource);



# await Import.FromFile("../Files/TransactionalData/Openings_CH_2020_12.csv").WithFormat(ImportFormats.Opening).WithTarget(DataSource).ExecuteAsync()

Import.FromFile("Files/NominalCashflows_CH_2020_12.xlsx", DataSource, format_=ImportFormats.Cashflow)

# await Import.FromFile("../Files/TransactionalData/Actuals_CH_2020_12.csv").WithFormat(ImportFormats.Actual).WithTarget(DataSource).ExecuteAsync()
# await Import.FromFile("../Files/TransactionalData/NominalCashflows_CH_2021_3.csv").WithFormat(ImportFormats.Cashflow).WithTarget(DataSource).ExecuteAsync()
# await Import.FromFile("../Files/TransactionalData/Actuals_CH_2021_3.csv").WithFormat(ImportFormats.Actual).WithTarget(DataSource).ExecuteAsync()
# await Import.FromFile("../Files/TransactionalData/SimpleValue_CH_2020_12.csv").WithFormat(ImportFormats.SimpleValue ).WithTarget(DataSource).ExecuteAsync()
# await Import.FromFile("../Files/TransactionalData/NominalCashflows_CH_2020_12_MTUP10pct.csv").WithFormat(ImportFormats.Cashflow).WithTarget(DataSource).ExecuteAsync()

# Workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions());


"""
Workspace.InitializeFrom(DataSource);
ifrs17.Reset(Workspace)
"""