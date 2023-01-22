from typing import (
    TypeVar, Generic, Iterable, Optional, Union, Callable, Any,
    Dict, Collection)
from collections import namedtuple
from .DataStructure import *
from .Extensions import *

# Current and Previous Parameters


def LoadCurrentParameterAsync(
    self,
    type_,
    args: Args,
    identityExpression: Callable[[IWithYearAndMonth], str],
    filterExpression: Callable[[IWithYearAndMonth], bool] = None) -> dict[str, IWithYearAndMonth]:

    temp = [x for x in self.LoadParameterAsync(type_, args.Year, args.Month, filterExpression)
             if x.Scenario == args.Scenario or x.Scenario == '']
    groupby = {}
    for x in temp:
        k = identityExpression(x)
        groupby.setdefault(k, []).append(x)

    return {k: max(v, key=lambda y: y.Year * 100 + y.Month, default=None)for k, v in groupby.items()}   # Scenario not considered


def LoadCurrentAndPreviousParameterAsync(
    self: IQuerySource,
    type_: type,
    args: Args,
    identityExpression: Callable[[IWithYearMonthAndScenario], str],
    filterExpression: Optional[Callable[[IWithYearMonthAndScenario], bool]] = None) -> dict[str, dict[int, IWithYearMonthAndScenario]]:

    parameters = {}
    for x in [yc for yc in self.LoadParameterAsync(type_, args.Year, args.Month, filterExpression)
              if yc.Scenario == args.Scenario or yc.Scenario == '']:
        
        k = identityExpression(x)
        parameters.setdefault(k, []).append(x)

    ret: dict[str, dict[int, IWithYearMonthAndScenario]] = {}

    for k, p in parameters.items():

        inner = ret.setdefault(k, {})
        currentCandidate = max([x for x in p if x.Year == args.Year], default=None, key= lambda y: y.Year * 100 + y.Month)
        previousCandidate = max([x for x in p if x.Year < args.Year and not x.Scenario], default=None, key= lambda y: y.Year * 100 + y.Month)
        currentCandidateBE = max([x for x in p if x.Year <= args.Year and not x.Scenario], default=None, key= lambda y: y.Year * 100 + y.Month)

        inner[CurrentPeriod] = currentCandidate if currentCandidate else previousCandidate
        inner[PreviousPeriod] = previousCandidate if previousCandidate else currentCandidateBE if currentCandidateBE else currentCandidate
        # TODO: log error if currentCandidate is null

    return ret

# Yield Curve


def LoadLockedInYieldCurveAsync(self: IQuerySource, args: Args, dataNodes: list[DataNodeData]) -> dict[str, YieldCurve]:

    lockedInYieldCurveByGoc: dict[str, YieldCurve] = dict()

    for dn in [x for x in dataNodes if x.ValuationApproach == ValuationApproaches.BBA]:

        kwargs = args.__dict__
        argsNew = type(args)(**kwargs)
        argsNew.Year = dn.Year
        argsNew.Month = dn.Month
        argsNew.Scenario = dn.Scenario

        loadedYc = self.LoadCurrentParameterAsync(YieldCurve, argsNew, lambda x: x.Currency, lambda x: x.Currency == dn.ContractualCurrency)

        if not (lockedYc := loadedYc.get(dn.ContractualCurrency, None)):
            raise YieldCurveNotFound

        lockedInYieldCurveByGoc[dn.DataNode] = lockedYc

    return lockedInYieldCurveByGoc



def LoadCurrentYieldCurveAsync(self,  args: Args, dataNodes: list[DataNodeData]) -> dict[str, dict[int, YieldCurve]]:
    contractualCurrenciesInScope = {dn.ContractualCurrency for dn in dataNodes}
    return self.LoadCurrentAndPreviousParameterAsync(YieldCurve, args,
            lambda x: x.Currency,
            lambda x: x.Currency in contractualCurrenciesInScope)


# Data Nodes

def LoadDataNodesAsync(self: IQuerySource, args: Args) -> dict[str, DataNodeData]:

    dataNodeStates = self.LoadCurrentAndPreviousParameterAsync(DataNodeState, args, lambda x: x.DataNode)
    activeDataNodes = [k for k, v in dataNodeStates.items() if v[CurrentPeriod].State != State.Inactive]

    temp = [dn for dn in self.Query(GroupOfContract) if dn.SystemName in activeDataNodes]
    result = {}
    for dn in temp:        
        dnCurrentState = dataNodeStates[dn.SystemName][CurrentPeriod]
        dnPreviousState = dataNodeStates[dn.SystemName][PreviousPeriod]
        result[dn.SystemName] = DataNodeData(Year=dnPreviousState.Year,
                                  Month=dnPreviousState.Month,
                                  State=dnCurrentState.State,
                                  PreviousState=dnPreviousState.State,
                                  DataNode=dn.SystemName,
                                  ContractualCurrency=dn.ContractualCurrency,
                                  FunctionalCurrency=dn.FunctionalCurrency,
                                  LineOfBusiness=dn.LineOfBusiness,
                                  ValuationApproach=dn.ValuationApproach,
                                  OciType=dn.OciType,
                                  Portfolio=dn.Portfolio,
                                  AnnualCohort=dn.AnnualCohort,
                                  LiabilityType=dn.LiabilityType,
                                  Profitability=dn.Profitability,
                                  Partner=dn.Partner,
                                  IsReinsurance=isinstance(dn, GroupOfReinsuranceContract),
                                  Scenario='')

    return result


def LoadParameterAsync(
    self,
    type_: type,
    year: int,
    month: int,
    filterExpression: Optional[Callable[[IWithYearAndMonth], bool]] = None) -> list[IWithYearAndMonth]:

    result = [x for x in self.Query(type_) if x.Year == year and x.Month <= month or x.Year < year]

    if filterExpression:
        result = [x for x in result if filterExpression(x)]

    return result


TempName1 = namedtuple('TempName1', ['key', 'currentPeriod', 'previousPeriod'])


@dataclass
class IGroupingParams:
    Key: Any
    Values: list[Any]


def LoadSingleDataNodeParametersAsync(self, args: Args) -> dict[str, dict[int, SingleDataNodeParameter]]:
    return self.LoadCurrentAndPreviousParameterAsync(SingleDataNodeParameter, args, lambda x: x.DataNode)


def LoadInterDataNodeParametersAsync(self, args: Args) -> dict[str, dict[int, set[InterDataNodeParameter]]]:

    identityExpressions: list[Callable[[InterDataNodeParameter], str]] = [lambda x: x.DataNode, lambda x: x.LinkedDataNode]
    parameterArray = self.LoadParameterAsync(InterDataNodeParameter, args.Year, args.Month)

    parameters = []
    for ie in identityExpressions:
        temp = {}
        for p in parameterArray:
            k = ie(p)
            if k in temp:
                temp[k].append(p)
            else:
                temp[k] = [p]
        parameters.extend([IGroupingParams(Key=key, Values=val) for key, val in temp.items()])

    def _inner(p, gg):
        currentCandidate = max([x for x in gg if x.Year == args.Year], key=lambda x: x.Month, default=None) #[-1]
        previousCandidate = max([x for x in gg if x.Year < args.Year], key=lambda x: x.Year * 100 + x.Month, default=None)  #[-1]
        return TempName1(key=p.Key,
             currentPeriod= currentCandidate if currentCandidate else previousCandidate,
             previousPeriod= previousCandidate if previousCandidate else currentCandidate)

    result: list[TempName1] = []
    for p in parameters:
        groups = {}
        for x in p.Values:
            key = x.DataNode if x.DataNode != p.Key else x.LinkedDataNode
            if key in groups:
                groups[key].append(x)
            else:
                groups[key] = [x]

        for gg in groups.values():
            result.append(_inner(p, gg))


    temp = {}
    for x in result:
        if x.key in temp:
            temp[x.key].append(x)
        else:
            temp[x.key] = [x]

    temp2: dict[str, dict[int, set[InterDataNodeParameter]]] = {}
    for k, v in temp.items():
        temp2[k] = {CurrentPeriod: set(y.currentPeriod for y in v),
                    PreviousPeriod: set(y.previousPeriod for y in v)}

    return temp2


def LoadAocStepConfigurationAsync(self, year: int, month: int) -> list[AocConfiguration]:

    temp = {}
    for x in self.LoadParameterAsync(AocConfiguration, year, month):
        key = (x.AocType, x.Novelty)
        temp.setdefault(key, []).append(x)

    result = []
    for k, v in temp.items():
        result.append(max(v, key=lambda y: y.Year * 100 + y.Month))

    return result


def LoadAocStepConfigurationAsDictionaryAsync(self, year: int, month: int) -> dict[AocStep, AocConfiguration]:
    return {AocStep(x.AocType, x.Novelty): x for x in self.LoadAocStepConfigurationAsync(year, month)}


IQuerySource.LoadLockedInYieldCurveAsync = LoadLockedInYieldCurveAsync
IQuerySource.LoadSingleDataNodeParametersAsync = LoadSingleDataNodeParametersAsync
IQuerySource.LoadInterDataNodeParametersAsync = LoadInterDataNodeParametersAsync
IQuerySource.LoadCurrentAndPreviousParameterAsync = LoadCurrentAndPreviousParameterAsync
IQuerySource.LoadDataNodesAsync = LoadDataNodesAsync
IQuerySource.LoadCurrentParameterAsync = LoadCurrentParameterAsync
IQuerySource.LoadCurrentYieldCurveAsync = LoadCurrentYieldCurveAsync
IQuerySource.LoadParameterAsync = LoadParameterAsync
IQuerySource.LoadAocStepConfigurationAsync = LoadAocStepConfigurationAsync
IQuerySource.LoadAocStepConfigurationAsDictionaryAsync = LoadAocStepConfigurationAsDictionaryAsync




