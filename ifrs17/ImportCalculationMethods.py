import math

from .DataStructure import *
from .Extensions import *




def ComputeDiscountAndCumulate(nominalValues: list[float], monthlyDiscounting: list[float], periodType: PeriodType) -> list[float]:

    if not nominalValues:
        return []   #Enumerable.Empty<double>().ToArray()

    ret = [0] * len(nominalValues)

    if periodType == PeriodType.BeginningOfPeriod:
        for i in range(len(nominalValues) - 1, -1, -1):    #(var i = nominalValues.Length - 1; i >= 0; i--)
                ret[i] = nominalValues[i] + (ret[i + 1] if i+1 < len(ret) else 0) * (monthlyDiscounting[int(i/12)] if int(i/12) < len(monthlyDiscounting) else 0)

        return ret

    for i in range(len(nominalValues) - 1, -1, -1):     # (var i = nominalValues.Length - 1; i >= 0; i--)
                ret[i] = (nominalValues[i] + (ret[i + 1] if i+1 < len(ret) else 0))  * (monthlyDiscounting[int(i/12)] if int(i/12) < len(monthlyDiscounting) else 0)

    return ret


def ComputeDiscountAndCumulateWithCreditDefaultRisk(nominalValues: list[float], monthlyDiscounting: list[float], nonPerformanceRiskRate: float) -> list[float]:

    #Is it correct that NonPerformanceRiskRate is a double? Should it be an array that takes as input tau/t?

    ret = []
    for t in range(len(nominalValues)):
        temp = []
        for tau in range(t, len(nominalValues) - t):
           temp.append(nominalValues[tau] * math.pow(GetElementOrDefault(monthlyDiscounting, int(t/12)), tau-t+1) * (math.exp(-nonPerformanceRiskRate*(tau-t)) - 1))

        ret.append(sum(temp))

    return ret


def GetReferenceAocStepForCalculated(identities: list[AocStep], aocConfigurationByAocStep: dict[AocStep, AocConfiguration], identityAocStep: AocStep) -> AocStep:

    for aocStep in reversed(identities):
        if (aocConfigurationByAocStep[aocStep].DataType != DataType.Calculated
                                            and aocConfigurationByAocStep[aocStep].DataType != DataType.CalculatedTelescopic
                                            and aocConfigurationByAocStep[aocStep].Order < aocConfigurationByAocStep[identityAocStep].Order
                                            and aocStep.Novelty == identityAocStep.Novelty):
            return aocStep

    return AocStep('', '')

        # ?? new AocStep(default, default);




def GetPreviousIdentities(identities: list[AocStep]) -> dict[AocStep, list[AocStep]]:

    bopNovelties = [id_.Novelty for id_ in identities if id_.AocType == AocTypes.BOP]   #.Where(id => id.AocType == AocTypes.BOP).Select(id_ => id_.Novelty)
    previousStep = {n: AocStep(AocTypes.BOP, n) if n in bopNovelties else None for n in [Novelties.N, Novelties.I, Novelties.C]}

    temp = {}
    for id_ in identities:
        if id_.AocType != AocTypes.BOP:
            ret = [v for v in previousStep.values() if v] if id_.AocType == AocTypes.CL else [previousStep[id_.Novelty]] if previousStep[id_.Novelty] else []   # previousStep.Where(kvp => kvp.Value != null).Select(kvp => kvp.Value).ToArray()
            previousStep[id_.Novelty] = AocStep(id_.AocType, id_.Novelty)
            temp[id_] = ret

    return temp


def ExtendGroupOfContract(gic: GroupOfContract, datarow: IDataRow) -> GroupOfContract:
    return gic


def GetAmountTypesByEstimateType(hierarchyCache: IHierarchicalDimensionCache) -> dict[str, set[str]]:
    return {
        EstimateTypes.RA: set(),
        EstimateTypes.C: set(),
        EstimateTypes.L: set(),
        EstimateTypes.LR: set()
   }


def GetTechnicalMarginEstimateType() -> set[str]:
    return {EstimateTypes.C, EstimateTypes.L, EstimateTypes.LR,}


def GetSign(variable: tuple, hierarchyCache: IHierarchicalDimensionCache) -> int:
    return 1
