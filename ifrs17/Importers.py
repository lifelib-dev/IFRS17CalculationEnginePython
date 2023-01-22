import dataclasses
import numpy as np
import pandas as pd
from . import Import
from .ImportScopeCalculation import *
from .ImportCalculationMethods import *


class ParsingStorage:

    # Hierarchy Cache

    HierarchyCache: IHierarchicalDimensionCache

    ReportingNode: ReportingNode
    DataNodeDataBySystemName: dict[str, DataNodeData]

    # Dimensions

    EstimateType: dict[str, EstimateType] 
    AmountType: dict[str, AmountType]

    AocTypeMap: set[AocStep] 
    estimateTypes: set[str] 
    amountTypes: set[str]

    @property
    def amountTypesByEstimateType(self) -> dict[str, set[str]]:
        return GetAmountTypesByEstimateType(self.HierarchyCache)

    @property
    def TechnicalMarginEstimateTypes(self) -> set[str]:
        return GetTechnicalMarginEstimateType()

    DimensionsWithExternalId: dict[type, dict[str, str]]

    # Partitions

    TargetPartitionByReportingNode: PartitionByReportingNode 
    TargetPartitionByReportingNodeAndPeriod: PartitionByReportingNodeAndPeriod 

    # Constructor
    def __init__(self, args: ImportArgs, dataSource: IDataSource, workspace: IWorkspace):
        self.args: ImportArgs = args
        self.dataSource: IDataSource = dataSource
        self.workspace: IWorkspace = workspace

    # Initialize
    def  InitializeAsync(self):

        # Partition Workspace and DataSource
        # TODO: Original code queries workspace instead. Don't know why.
        node_ = [p for p in self.dataSource.Query(PartitionByReportingNode) if p.ReportingNode == self.args.ReportingNode]

        self.TargetPartitionByReportingNode = node_[0] if len(node_) else None

        if not self.TargetPartitionByReportingNode:
            raise ParsedPartitionNotFound

        self.workspace.Partition.SetAsync(PartitionByReportingNode, self.TargetPartitionByReportingNode.Id)
        self.dataSource.Partition.SetAsync(PartitionByReportingNode, self.TargetPartitionByReportingNode.Id)

        if self.args.Year != 0 and self.args.Month != 0:

            node_ = [p for p in self.dataSource.Query(PartitionByReportingNodeAndPeriod)    # TODO: Original code queries workspace instead. Don't know why.
                if p.ReportingNode == self.args.ReportingNode and
                p.Year == self.args.Year and
                p.Month == self.args.Month and
                p.Scenario == self.args.Scenario]

            self.TargetPartitionByReportingNodeAndPeriod = node_[0] if len(node_) else None

            if self.TargetPartitionByReportingNodeAndPeriod is None:
                raise ParsedPartitionNotFound

            self.workspace.Partition.SetAsync(PartitionByReportingNodeAndPeriod, self.TargetPartitionByReportingNodeAndPeriod.Id)
            self.dataSource.Partition.SetAsync(PartitionByReportingNodeAndPeriod, self.TargetPartitionByReportingNodeAndPeriod.Id)

            #Clean up the workspace
            self.workspace.DeleteAsync(RawVariable, self.workspace.Query(RawVariable))
            self.workspace.DeleteAsync(IfrsVariable, self.workspace.Query(IfrsVariable))

        reportingNodes = [x for x in self.dataSource.Query(ReportingNode) if x.SystemName == self.args.ReportingNode]
        if not reportingNodes:
            raise ReportingNodeNotFound

        self.ReportingNode = reportingNodes[0]

        aocConfigurationByAocStep = self.dataSource.LoadAocStepConfigurationAsync(self.args.Year, self.args.Month)

        if self.args.ImportFormat == ImportFormats.Cashflow:

            self.AocTypeMap = set(AocStep(x.AocType, x.Novelty) for x in aocConfigurationByAocStep
                            if (InputSource.Cashflow in x.InputSource
                                and not x.DataType in (DataType.Calculated,
                                                       DataType.CalculatedTelescopic)))

        elif self.args.ImportFormat == ImportFormats.Actual:

            self.AocTypeMap = set(AocStep(x.AocType, x.Novelty) for x in aocConfigurationByAocStep
                               if (InputSource.Cashflow in x.InputSource
                                and not x.DataType in (DataType.Calculated,
                                                       DataType.CalculatedTelescopic)
                                   and (x.AocType, x.Novelty) != (AocTypes.BOP, Novelties.I)))

        elif self.args.ImportFormat == ImportFormats.Opening:

            self.AocTypeMap = set(AocStep(x.AocType, x.Novelty) for x in aocConfigurationByAocStep
                            if (InputSource.Opening in x.InputSource
                                and x.DataType in DataType.Optional))

        elif self.args.ImportFormat == ImportFormats.SimpleValue:

            self.AocTypeMap = (
                    set(AocStep(x.AocType, x.Novelty) for x in aocConfigurationByAocStep) |
                    set(AocStep(vt.SystemName, '') for vt in self.dataSource.Query(PnlVariableType)))
        else:
            self.AocTypeMap = set()


        # DataNodes

        if self.args.ImportFormat == ImportFormats.Opening:
            self.DataNodeDataBySystemName = {k: v for k, v in self.dataSource.LoadDataNodesAsync(self.args).items() if v.Year == self.args.Year}
        else:
            self.DataNodeDataBySystemName = self.dataSource.LoadDataNodesAsync(self.args)

        # Dimensions

        self.EstimateType = {x.SystemName: x for x in self.dataSource.Query(EstimateType)}
        self.AmountType = {x.SystemName: x for x in self.dataSource.Query(AmountType) if not isinstance(x, DeferrableAmountType)}
        self.amountTypes = set(x.SystemName for x in self.dataSource.Query(AmountType))

        if self.args.ImportFormat == ImportFormats.SimpleValue:
            self.estimateTypes = set(et.SystemName for et in self.dataSource.Query(EstimateType))
        elif self.args.ImportFormat == ImportFormats.Opening:
            self.estimateTypes = set(et.SystemName for et in self.dataSource.Query(EstimateType)
                                     if et.StructureType == StructureType.AoC and
                                     InputSource.Opening in et.InputSource)
        else:
            self.estimateTypes = set()

        # DimensionsWithExternalId

        self.DimensionsWithExternalId = {
            type(self.AmountType): self.GetDimensionWithExternalIdDictionaryAsync(AmountType),
            type(self.EstimateType): self.GetDimensionWithExternalIdDictionaryAsync(EstimateType)
        }

        # # Hierarchy Cache
        self.HierarchyCache = self.workspace.ToHierarchicalDimensionCache()
        self.HierarchyCache.InitializeAsync(AmountType)

    def GetDimensionWithExternalIdDictionaryAsync(self, T: type) -> dict[str, str]: # T = KeyedOrderedDimension

        dict_ = {}
        items = self.dataSource.Query(T)
        for item in items:
            if item.SystemName not in dict_:
                dict_[item.SystemName] = item.SystemName

            if issubclass(T, KeyedOrderedDimensionWithExternalId):
                externalIds = item.ExternalId
                if not externalIds:
                    continue
                    
                for extId in externalIds:
                    if item.SystemName not in dict_:
                        dict_[extId] = item.SystemName

        return dict_

    # Getters

    def IsDataNodeReinsurance(self, goc: str) -> bool:
        return self.DataNodeDataBySystemName[goc].IsReinsurance

    def IsValidDataNode(self, goc: str) -> bool:
        return goc in self.DataNodeDataBySystemName

    # Validations

    def ValidateEstimateType(self, et: str, goc: str) -> str:

        allowedEstimateTypes = self.estimateTypes
        dataNodeData = self.DataNodeDataBySystemName.get(goc, None)

        if dataNodeData and dataNodeData.LiabilityType == LiabilityTypes.LIC:
            for elm in self.TechnicalMarginEstimateTypes:
                self.estimateTypes.remove(elm)

        if et in allowedEstimateTypes:
            raise EstimateTypeNotFound

        return et

    def ValidateAmountType(self, at: str) -> str:
        if at and at not in self.amountTypes:
            raise AmountTypeNotFound

        return at

    def ValidateAocStep(self, aoc: AocStep) -> AocStep:
        if aoc not in self.AocTypeMap:
            raise AocTypeMapNotFound

        return aoc

    def ValidateDataNode(self, goc: str) -> str:
        if goc not in self.DataNodeDataBySystemName:
            raise InvalidDataNode
        return goc

    def ValidateEstimateTypeAndAmountType(self, estimateType: str, amountType: str):
        ats = self.amountTypesByEstimateType.get(estimateType, None)
        if ats and any(ats) and amountType not in ats:
            raise InvalidAmountTypeEstimateType


## Update the Database

def CommitToDatabase(
        T: type,    #IPartitioned
        workspace: IWorkspace, partitionId: Guid, snapshot: bool=True, filter: Callable[IPartitioned, bool] = None):

    # if(snapshot) CleanDatabaseFromPartitionAsync<T>(partitionId, filter)
    DataSource.UpdateAsync(T, workspace.Query(T))
    # DataSource.CommitAsync()


# Import helper


def GetArgsFromMainAsync(PartitionType: IKeyedType, dataSet: IDataSet) -> ImportArgs:

    mainTab: pd.DataFrame = dataSet.Tables[Main]

    # if(mainTab == null) ApplicationMessage.Log(Error.NoMainTab)
    # if(mainTab.Rows.Count() == 0) ApplicationMessage.Log(Error.IncompleteMainTab)
    # if(mainTab.Columns.FirstOrDefault(x => x.ColumnName == nameof(ReportingNode)) == null) ApplicationMessage.Log(Error.ReportingNodeInMainNotFound)
    # if(ApplicationMessage.HasErrors()) return null

    main = mainTab.iloc[0]  # mainTab.Rows.First()
    reportingNode = main["ReportingNode"]
    scenario = main["Scenario"] if 'Scenario' in mainTab.columns and not np.isnan(main["Scenario"]) else ""   # Convert nan to ''

    args: ImportArgs

    if PartitionType is PartitionByReportingNode:
        args = ImportArgs(reportingNode, 0, 0, Periodicity.Monthly, scenario, "")
        DataSource.UpdateAsync(PartitionByReportingNode,
            [PartitionByReportingNode(
                         Id=DataSource.Partition.GetKeyForInstanceAsync(PartitionByReportingNode, args),
                         ReportingNode=reportingNode,
                         Scenario=scenario)])

    elif PartitionType is PartitionByReportingNodeAndPeriod:

        if list(mainTab.columns).count('Year') != 1:
            raise YearInMainNotFound
        if list(mainTab.columns).count('Month') != 1:
            raise MonthInMainNotFound

        args = ImportArgs(reportingNode, int(main["Year"]), int(main["Month"]), Periodicity.Monthly, scenario, "")

        DataSource.UpdateAsync(PartitionByReportingNodeAndPeriod,
          [PartitionByReportingNodeAndPeriod(
                Id=DataSource.Partition.GetKeyForInstanceAsync(PartitionByReportingNodeAndPeriod, args),
                Year=args.Year,
                Month=args.Month,
                ReportingNode=reportingNode,
                Scenario=scenario)])

    else:
        # ApplicationMessage.Log(Error.PartitionTypeNotFound, typeof(IPartition).Name)
        raise PartitionTypeNotFound

    # await DataSource.CommitAsync()
    return args


## Data Node Factory


def DataNodeFactoryAsync(dataSet: IDataSet, tableName: str, args: ImportArgs):

    # Debug
    # result = []
    # for p in DataSource.Query(PartitionByReportingNode):
    #     if p.ReportingNode == args.ReportingNode and p.Scenario == '':
    #         result.append(p)
    
    partition = [p for p in DataSource.Query(PartitionByReportingNode)
                 if p.ReportingNode == args.ReportingNode and p.Scenario == ''][0]

    if not partition:
        raise ParsedPartitionNotFound

    table = dataSet.Tables[tableName]

    dataNodesImported = set(table["DataNode"])
    dataNodesDefined = [x for x in DataSource.Query(GroupOfContract) if x.SystemName in dataNodesImported]
    dataNodeStatesDefined = [x.DataNode for x in DataSource.Query(DataNodeState)]
    dataNodeParametersDefined = [x.DataNode for x in DataSource.Query(SingleDataNodeParameter)]

    dataNodeStatesUndefined = set(x for x in dataNodesImported if x and not x in dataNodeStatesDefined)
    dataNodeSingleParametersUndefined = set(x for x in dataNodesImported if (x and
                                                                    not x in dataNodeParametersDefined and
                                                                    isinstance([y for y in dataNodesDefined if y.SystemName == x][0], GroupOfInsuranceContract)))

    DataSource.UpdateAsync3(
        [DataNodeState(DataNode=x,
                       Year=args.Year,
                       Month=DefaultDataNodeActivationMonth,
                       State=State.Active,
                       Partition=partition.Id,
                       Id=uuid.uuid4(),
                       Scenario=''
                       ) for x in dataNodeStatesUndefined]
        )

    DataSource.UpdateAsync3( 
        [SingleDataNodeParameter(
            DataNode=x, Year=args.Year,
            Month=DefaultDataNodeActivationMonth,
            PremiumAllocation=DefaultPremiumExperienceAdjustmentFactor,
            Partition=partition.Id,
            Id=uuid.uuid4(),
            Scenario=''
        ) for x in dataNodeSingleParametersUndefined]
    )
    # DataSource.CommitAsync()


# AocConfiguration

def _DefineFormatAocConfiguration(options: Import.Options, dataSet: IDataSet):

    s_to_i = {k: v for k, v in InputSource.__dict__.items() if k[0] != '_'}

    dataSet.Tables['AocConfiguration']['InputSource'] = dataSet.Tables['AocConfiguration']['InputSource'].apply(lambda x: [s_to_i[s.strip()] for s in x.split(',')])

    workspace = IWorkspace()
    workspace.InitializeFrom(options.TargetDataSource)

    aocTypes = sorted(options.TargetDataSource.Query(AocType), key=lambda x: x.Order)
    aocTypesCompulsory = [v for k, v in AocTypes.__dict__.items() if k[0] != '_']

    if any(x not in [y.SystemName for y in aocTypes] for x in aocTypesCompulsory):
        raise AocTypeCompulsoryNotFound

    Import.FromDataSet(dataSet, AocConfiguration, workspace)

    # if(logConfig.Errors.Any()) return Activity.Finish().Merge(logConfig)

    orderByName = {x.SystemName: x.Order for x in aocTypes}

    temp: dict[(AocTypes, Novelties), list[AocConfiguration]] = {}
    for x in workspace.Query(AocConfiguration):
        k = (x.AocType, x.Novelty)
        if k in temp:
            temp[k].append(x)
        else:
            temp[k] = [x]

    aocConfigs: dict[(AocTypes, Novelties), AocConfiguration] = {}
    for k, v in temp.items():
        aocConfigs[k] = max(v, key=lambda x: x.Year * 100 + x.Month)

    aocOrder: dict[(AocTypes, Novelties), int] = {k: v.Order for k, v in aocConfigs.items()}

    newAoCTypes: list[str] = [x for x in orderByName.keys() if (
        (x, Novelties.I) not in aocConfigs.keys()
        and (x, Novelties.N) not in aocConfigs.keys()
        and (x, Novelties.C) not in aocConfigs.keys()
        and not any(y.Parent == x for y in aocTypes)
        and not x in aocTypesCompulsory
    )]

    for newAocType in newAoCTypes:

        if orderByName[newAocType] < orderByName[AocTypes.RCU]:
            step = (AocTypes.MC, Novelties.I)

            temp2: AocConfiguration = dataclasses.replace(aocConfigs[step])
            temp2.AocType = newAocType
            temp2.DataType = DataType.Optional
            temp2.Order = aocOrder[step] + 1
            workspace.UpdateAsync2(temp2)

        elif (orderByName[newAocType] > orderByName[AocTypes.RCU] and orderByName[newAocType] < orderByName[AocTypes.CF]):
            
            step = (AocTypes.RCU, Novelties.I)

            temp2: AocConfiguration = dataclasses.replace(aocConfigs[step])
            temp2.AocType = newAocType
            temp2.DataType = DataType.Optional
            temp2.Order = aocOrder[step] + 1
            workspace.UpdateAsync2(temp2)

        elif (orderByName[newAocType] > orderByName[AocTypes.IA] and orderByName[newAocType] < orderByName[AocTypes.YCU]):
            
            for novelty in (Novelties.I, Novelties.N):
                step = (AocTypes.AU, novelty)
                order = aocOrder[(AocTypes.IA, novelty)] + 1 if orderByName[newAocType] < orderByName[AocTypes.AU] else aocOrder[(AocTypes.AU, novelty)] + 1
                
                temp2: AocConfiguration = dataclasses.replace(aocConfigs[step])
                temp2.AocType = newAocType
                temp2.DataType = DataType.Optional
                temp2.Order = order
                workspace.UpdateAsync2(temp2)

        elif (orderByName[newAocType] > orderByName[AocTypes.CRU] and orderByName[newAocType] < orderByName[AocTypes.WO]):
            
            stepI = (AocTypes.EV, Novelties.I)
            orderI = aocOrder[(AocTypes.CRU, Novelties.I)] + 1 if orderByName[newAocType] < orderByName[AocTypes.EV] else aocOrder[(AocTypes.EV, Novelties.I)] + 1

            temp2: AocConfiguration = dataclasses.replace(aocConfigs[stepI])
            temp2.AocType = newAocType
            temp2.DataType = DataType.Optional
            temp2.Order = orderI
            workspace.UpdateAsync2(temp2)

            stepN = (AocTypes.EV, Novelties.N)
            orderN = aocOrder[(AocTypes.AU, Novelties.N)] + 1 if orderByName[newAocType] < orderByName[AocTypes.EV] else aocOrder[(AocTypes.EV, Novelties.N)] + 1

            temp2: AocConfiguration = dataclasses.replace(aocConfigs[stepN])
            temp2.AocType = newAocType
            temp2.DataType = DataType.Optional
            temp2.Order = orderN
            workspace.UpdateAsync2(temp2)

        elif (orderByName[newAocType] > orderByName[AocTypes.WO] and orderByName[newAocType] < orderByName[AocTypes.CL]):

            step = (AocTypes.WO, Novelties.C)

            temp2: AocConfiguration = dataclasses.replace(aocConfigs[step])
            temp2.AocType = newAocType
            temp2.DataType = DataType.Optional
            temp2.Order = aocOrder[step] + 1
            workspace.UpdateAsync2(temp2)

        else:
            raise AocTypePositionNotSupported

    # var aocConfigsFinal = await workspace.Query<AocConfiguration>().ToArrayAsync();
    #
    # if(aocConfigsFinal.GroupBy(x => x.Order).Any(x => x.Count() > 1))
    #     ApplicationMessage.Log(Error.AocConfigurationOrderNotUnique);

    workspace.CommitToTargetAsync(options.TargetDataSource)

    # return Activity.Finish().Merge(logConfig);


Import.DefineFormat(ImportFormats.AocConfiguration, _DefineFormatAocConfiguration)


# Data Nodes

def UploadDataNodesToWorkspaceAsync(dataSet: IDataSet, workspace: IWorkspace):

    # workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions())
    workspace.Initialize(DataSource, DisableInitialization=[RawVariable, IfrsVariable, DataNodeState, DataNodeParameter])

    # Activity.Start()
    args = GetArgsFromMainAsync(PartitionByReportingNode, dataSet)
    # if(Activity.HasErrors()) return Activity.Finish()

    storage = ParsingStorage(args, DataSource, workspace)
    storage.InitializeAsync()
    # if(Activity.HasErrors()) return Activity.Finish()

    # var errors = new List<string>()
    
    def _FromDataSetInsurancePortfolio(dataSet, datarow):
        return InsurancePortfolio(
            SystemName=datarow["SystemName"],
            DisplayName=datarow["DisplayName"],
            Partition=storage.TargetPartitionByReportingNode.Id,
            ContractualCurrency=datarow["ContractualCurrency"],
            FunctionalCurrency=storage.ReportingNode.Currency,
            LineOfBusiness=datarow["LineOfBusiness"],
            ValuationApproach=datarow["ValuationApproach"],
            OciType=datarow["OciType"]
        )
                                                                                    
    importLogPortfolios = Import.FromDataSet(
        dataSet, 
        type_=InsurancePortfolio,
        workspace=workspace,
        body=_FromDataSetInsurancePortfolio)
    
    def _FromDataSetReinsurancePortfolio(dataSet, datarow):
        return ReinsurancePortfolio(
            SystemName=datarow["SystemName"],
            DisplayName=datarow["DisplayName"],
            Partition=storage.TargetPartitionByReportingNode.Id,
            ContractualCurrency=datarow["ContractualCurrency"],
            FunctionalCurrency=storage.ReportingNode.Currency,
            LineOfBusiness=datarow["LineOfBusiness"],
            ValuationApproach=datarow["ValuationApproach"],
            OciType=datarow["OciType"]
        )

    if ReinsurancePortfolio.__name__ in dataSet.Tables:

        Import.FromDataSet(
            dataSet,
            type_=ReinsurancePortfolio,
            workspace=workspace,
            body=_FromDataSetReinsurancePortfolio)

    portfolios = {x.SystemName: x for x in workspace.Query(Portfolio)}
    
    def _FromDataSetGroupOfContracts(dataset, datarow):

        gicSystemName = datarow["SystemName"]
        pf = datarow["InsurancePortfolio"]

        portfolioData = portfolios.get(pf, None)

        if not portfolioData:
            raise PortfolioGicNotFound

        gic = GroupOfInsuranceContract(
            SystemName=gicSystemName,
            DisplayName=datarow["DisplayName"],
            Partition=storage.TargetPartitionByReportingNode.Id,
            ContractualCurrency=portfolioData.ContractualCurrency,
            FunctionalCurrency=portfolioData.FunctionalCurrency,
            LineOfBusiness=portfolioData.LineOfBusiness,
            ValuationApproach=portfolioData.ValuationApproach,
            OciType=portfolioData.OciType,
            AnnualCohort= int(datarow["AnnualCohort"]),
            LiabilityType=datarow["LiabilityType"],
            Profitability=datarow["Profitability"],
            Portfolio=pf,
            YieldCurveName=datarow["YieldCurveName"] if "YieldCurveName" in dataset.Tables["GroupOfInsuranceContract"].columns else '',
            Partner=''
        )

        return ExtendGroupOfContract(gic, datarow)

    def _FromDataSetGroupOfReinsuranceContract(dataset, datarow):

        gricSystemName = datarow["SystemName"]
        pf = datarow["ReinsurancePortfolio"]
        
        portfolioData = portfolios.get(pf, None)
        if not portfolioData:
            raise PortfolioGicNotFound
        
        gric = GroupOfReinsuranceContract(
    
            SystemName=gricSystemName,
            DisplayName=datarow["DisplayName"],
            Partition=storage.TargetPartitionByReportingNode.Id,
            ContractualCurrency=portfolioData.ContractualCurrency,
            FunctionalCurrency=portfolioData.FunctionalCurrency,
            LineOfBusiness=portfolioData.LineOfBusiness,
            ValuationApproach=portfolioData.ValuationApproach,
            OciType=portfolioData.OciType,
            AnnualCohort=int(datarow["AnnualCohort"]),
            LiabilityType=datarow["LiabilityType"],
            Profitability=datarow["Profitability"],
            Portfolio=pf,
            Partner=datarow["Partner"],
            YieldCurveName=datarow["YieldCurveName"] if "YieldCurveName" in dataset.Tables["GroupOfInsuranceContract"].columns.values else '',

        )
        return ExtendGroupOfContract(gric, datarow)
    
    importLogGroupOfContracts = Import.FromDataSet(dataSet, GroupOfInsuranceContract, workspace, _FromDataSetGroupOfContracts)
    if "GroupOfReinsuranceContract" in dataSet.Tables:
        importLogGroupOfContracts = Import.FromDataSet(dataSet, GroupOfReinsuranceContract, workspace, _FromDataSetGroupOfReinsuranceContract)

    # return Activity.Finish().Merge(importLogPortfolios).Merge(importLogGroupOfContracts)


def _DefineFormatDataNode(options: Import.Options, dataSet: IDataSet):

    workspace = IWorkspace()
    log =  UploadDataNodesToWorkspaceAsync(dataSet, workspace)

    partition = workspace.Partition.GetCurrent("PartitionByReportingNode")

    CommitToDatabase(InsurancePortfolio, workspace, partition)
    CommitToDatabase(ReinsurancePortfolio, workspace, partition)
    CommitToDatabase(GroupOfInsuranceContract, workspace, partition)
    CommitToDatabase(GroupOfReinsuranceContract, workspace, partition)
    # return log


Import.DefineFormat(ImportFormats.DataNode, _DefineFormatDataNode)


# Data Node State


def UploadDataNodeStateToWorkspaceAsync(dataSet: IDataSet, workspace: IWorkspace):

    # workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions())
    workspace.Initialize(DataSource, DisableInitialization=[RawVariable, IfrsVariable, DataNodeState])

    args = GetArgsFromMainAsync(PartitionByReportingNodeAndPeriod, dataSet)

    # if(Activity.HasErrors()) return Activity.Finish();

    storage = ParsingStorage(args, DataSource, workspace)
    storage.InitializeAsync()

    # if(Activity.HasErrors()) return Activity.Finish();

    importLog = Import.FromDataSet(dataSet, DataNodeState, workspace,

        lambda dataset, datarow: DataNodeState(
            Id=Guid(bytes=b'\0'*16),
            DataNode=datarow["DataNode"],
            State=datarow["State"],
            Year=args.Year,
            Month=args.Month,
            Partition=storage.TargetPartitionByReportingNode.Id,
            Scenario=''
        )
    )
    # await workspace.ValidateDataNodeStatesAsync(storage.DataNodeDataBySystemName)
    # return Activity.Finish().Merge(importLog)


def _DefineFormatDataNodeState(options: Import.Options, dataSet: IDataSet):

    workspace = IWorkspace()
    log = UploadDataNodeStateToWorkspaceAsync(dataSet, workspace)

    CommitToDatabase(DataNodeState, workspace, workspace.Partition.GetCurrent("PartitionByReportingNode"), snapshot=False)
    return log



Import.DefineFormat(ImportFormats.DataNodeState, _DefineFormatDataNodeState)


# Data Node Parameters

def UploadDataNodeParameterToWorkspaceAsync(dataSet: IDataSet, targetPartitionByReportingNodeId: Guid, workspace: IWorkspace):

    # workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions())
    workspace.Initialize(DataSource, DisableInitialization=[RawVariable, IfrsVariable, DataNodeParameter])

    # Activity.Start()
    args = GetArgsFromMainAsync(PartitionByReportingNodeAndPeriod, dataSet)
    args.ImportFormat = ImportFormats.DataNodeParameter
    # if(Activity.HasErrors()) return Activity.Finish()

    storage = ParsingStorage(args, DataSource, workspace)
    storage.InitializeAsync()
    # if(Activity.HasErrors()) return Activity.Finish()

    singleDataNode = []     #new List<string>()
    interDataNode = []      #new List<(string,string)>()

    def _FromDataSetSingleDataNodeParameter(dataset, datarow):
        
        # read and validate DataNodes
        dataNode = datarow["DataNode"]
        if not storage.IsValidDataNode(dataNode):
            raise InvalidDataNode
        
        # check for duplicates
        if dataNode in singleDataNode:
           raise DuplicateSingleDataNode

        singleDataNode.append(dataNode)

        # Instantiate SingleDataNodeParameter

        return SingleDataNodeParameter(
            Id=uuid.uuid4(),
            Year=args.Year,
            Month=args.Month,
            Partition=storage.TargetPartitionByReportingNode.Id,
            DataNode=dataNode,
            PremiumAllocation=datarow["PremiumAllocation"],
            Scenario=''
        )

    def _FromDataSetInterDataNodeParameter(dataset, datarow):

        # read and validate DataNodes

        dataNode = datarow["DataNode"]
        if not storage.IsValidDataNode(dataNode):
            raise InvalidDataNode

        linkedDataNode = datarow["LinkedDataNode"]
        if not storage.IsValidDataNode(linkedDataNode):
            raise InvalidDataNode

        dataNodes = sorted([dataNode, linkedDataNode])

        # validate ReinsuranceGross Link

        isDn1Reinsurance = storage.IsDataNodeReinsurance(dataNodes[0])
        isDn2Reinsurance = storage.IsDataNodeReinsurance(dataNodes[1])
        isGrossReinsuranceLink = (isDn1Reinsurance and not isDn2Reinsurance) != (not isDn1Reinsurance and isDn2Reinsurance)
        reinsCov = datarow["ReinsuranceCoverage"]

        if(not isGrossReinsuranceLink and abs(reinsCov) > Precision):
            raise ReinsuranceCoverageDataNode

        # check for duplicates
        if ((dataNodes[0], dataNodes[1]) in interDataNode or (dataNodes[1], dataNodes[0]) in interDataNode):
            raise DuplicateInterDataNode

        interDataNode.append((dataNodes[0], dataNodes[1]))

        # Instantiate InterDataNodeParameter
        return InterDataNodeParameter(
            Id=uuid.uuid4(),
            Year=args.Year,
            Month=args.Month,
            Partition=storage.TargetPartitionByReportingNode.Id,
            DataNode=dataNodes[0],
            LinkedDataNode=dataNodes[1],
            ReinsuranceCoverage=reinsCov,
            Scenario=''
        )

    importLog = Import.FromDataSet(dataSet, SingleDataNodeParameter, workspace, _FromDataSetSingleDataNodeParameter)
    importLog = Import.FromDataSet(dataSet, InterDataNodeParameter, workspace, _FromDataSetInterDataNodeParameter)

    # targetPartitionByReportingNodeId = storage.TargetPartitionByReportingNode.Id
    # return Activity.Finish().Merge(importLog)


def _DefineFormatDataNodeParameter(options: Import.Options, dataSet: IDataSet):

    partitionId: Guid = uuid.uuid4()
    workspace = IWorkspace()
    log = UploadDataNodeParameterToWorkspaceAsync(dataSet, partitionId, workspace)
    CommitToDatabase(SingleDataNodeParameter, workspace, partitionId, snapshot=False)
    CommitToDatabase(InterDataNodeParameter, workspace, partitionId, snapshot=False)
    return log


Import.DefineFormat(ImportFormats.DataNodeParameter, _DefineFormatDataNodeParameter)


# Cashflows


def ParseCashflowsToWorkspaceAsync(dataSet: IDataSet, args: ImportArgs, workspace: IWorkspace) -> ActivityLog:

    # workspace.Reset(x => x.ResetInitializationRules().ResetCurrentPartitions())
    workspace.Initialize(DataSource, [RawVariable, IfrsVariable])

    # Activity.Start()
    parsingStorage = ParsingStorage(args, DataSource, workspace)
    parsingStorage.InitializeAsync()
    # if(Activity.HasErrors()) return Activity.Finish()

    def _FromDataSetCashflow(dataset, datarow):

            aocType = datarow["AocType"]
            novelty = datarow["Novelty"]
            dataNode = datarow["DataNode"]

            dataNodeData = parsingStorage.DataNodeDataBySystemName.get(dataNode, None)

            if not dataNodeData:
                raise InvalidDataNode

            # Error if AocType is not present in the mapping
            if AocStep(aocType, novelty) not in parsingStorage.AocTypeMap:
                raise AocTypeMapNotFound

            # Filter out cash flows for DataNode that were created in the past and are still active and come with AocType = BOPI

            if dataNodeData.Year < args.Year and aocType == AocTypes.BOP and novelty == Novelties.I:
                raise RuntimeError("ActiveDataNodeWithCashflowBOPI")

            amountTypeFromFile = datarow["AmountType"]
            isEstimateType = amountTypeFromFile in parsingStorage.EstimateType
            amountType =  None if isEstimateType else amountTypeFromFile
            estimateType =  amountTypeFromFile if isEstimateType else EstimateTypes.BE
            
            values = []
            for k, v in datarow.items():
                if k[:6] == "Values":
                    assert len(values) == int(k[6:])    # Check if Values are in ascending order.
                    values.append(float(v))

            # Filter out empty raw variables for AocType != CL#
            if len(values) == 0 and aocType != AocTypes.CL:
                return None  #TODO: extend this check for all mandatory step and not just for CL

            if len([x for x in dataset.Tables[ImportFormats.Cashflow].columns if x == "AccidentYear"]) > 0:
                temp = datarow["AccidentYear"]
            else:
                temp = None

            item = RawVariable(
                Id=uuid.uuid4(),
                DataNode=dataNode,
                AocType=aocType,
                Novelty=novelty,
                AmountType=amountType,
                EstimateType=estimateType,
                AccidentYear=temp,
                Partition=parsingStorage.TargetPartitionByReportingNodeAndPeriod.Id,
                Values=GetSign((aocType, amountType, estimateType, dataNodeData.IsReinsurance), parsingStorage.HierarchyCache) * values
            )
            return item


    importLog = Import.FromDataSet(dataSet, RawVariable, workspace, _FromDataSetCashflow, format_=ImportFormats.Cashflow)

    # await ValidateForDataNodeStateActiveAsync<RawVariable>(workspace, parsingStorage.DataNodeDataBySystemName)
    # return Activity.Finish().Merge(importLog)


def _DefineFormatCashflow(options, dataSet: IDataSet):

    # Activity.Start()
    args = GetArgsFromMainAsync(PartitionByReportingNodeAndPeriod, dataSet)
    args.ImportFormat = ImportFormats.Cashflow
    DataNodeFactoryAsync(dataSet, ImportFormats.Cashflow, args)
    # if(Activity.HasErrors()) return Activity.Finish()

    workspace = IWorkspace()
    parsingLog = ParseCashflowsToWorkspaceAsync(dataSet, args, workspace)
    # if(parsingLog.Errors.Any()) return Activity.Finish().Merge(parsingLog)

    storage = ImportStorage(args, DataSource, workspace)
    storage.InitializeAsync()
    # if(Activity.HasErrors()) return Activity.Finish().Merge(parsingLog)

    universe = IModel(storage)         # Scopes.ForStorage(storage).ToScope<IModel>()
    identities = sorted([i for s in universe.GetScopes(GetIdentities, storage.DataNodesByImportScope[ImportScope.Primary]) for i in s.Identities])

    # # For debug
    # identities = []
    # for s in universe.GetScopes(GetIdentities, storage.DataNodesByImportScope[ImportScope.Primary]):
    #     for i in s.Identities:
    #         identities.append(i)

    # For debug
    # ivs = []
    # scopes = universe.GetScopes(ComputeIfrsVarsCashflows, identities)
    # for i, s in enumerate(scopes):
    #     for x in s.CalculatedIfrsVariables:
    #         ivs.append(x)

    ivs = [x for s in universe.GetScopes(ComputeIfrsVarsCashflows, identities) for x in s.CalculatedIfrsVariables]

    # # For debug
    # ivs = []
    # for s in universe.GetScopes(ComputeIfrsVarsCashflows, identities):
    #     for x in s.CalculatedIfrsVariables:
    #         ivs.append(x)

    # if(Activity.HasErrors()) return Activity.Finish().Merge(parsingLog)

    workspace.UpdateAsync(IfrsVariable, ivs)
    CommitToDatabase(IfrsVariable, workspace,
                         storage.TargetPartition,
                         snapshot=True,
                         filter= lambda x: x.EstimateType in storage.EstimateTypesByImportFormat[ImportFormats.Cashflow]
                                           and x.DataNode in storage.DataNodesByImportScope[ImportScope.Primary])
    CommitToDatabase(RawVariable, workspace,
                        storage.TargetPartition,
                        snapshot=True,
                        filter= lambda x: x.DataNode in storage.DataNodesByImportScope[ImportScope.Primary])

    # return Activity.Finish().Merge(parsingLog)


Import.DefineFormat(ImportFormats.Cashflow, _DefineFormatCashflow)


# Actuals

def ParseActualsToWorkspaceAsync(dataSet: IDataSet, args: ImportArgs, workspace: IWorkspace):
    pass

def _DefineFormatActual(options, dataSet: IDataSet):
    pass

Import.DefineFormat(ImportFormats.Actual, _DefineFormatActual)

# Simple Value

def ParseSimpleValueToWorkspaceAsync(dataSet: IDataSet, args: ImportArgs, targetPartitionByReportingNodeAndPeriodId: Guid, workspace: IWorkspace):
    pass

def _DefineFormatSimpleValue(options, dataSet: IDataSet):
    pass

Import.DefineFormat(ImportFormats.SimpleValue, _DefineFormatSimpleValue)

# Opening

def _DefineFormatOpening(options, dataSet: IDataSet):
    pass

Import.DefineFormat(ImportFormats.Opening, _DefineFormatOpening)

