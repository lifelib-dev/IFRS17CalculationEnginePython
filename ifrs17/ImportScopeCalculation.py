import dataclasses
import math
from typing import TypeVar, Generic
from dataclasses import dataclass
from .ImportStorage import *
from .Extensions import *
from .ImportCalculationMethods import *

class IModel:

    def __init__(self, storage: ImportStorage):
        self.storage = storage

    def GetScopes(self, type_: type, arglist: Collection[Union[str, ImportIdentity]]):
        result = []
        for arg in arglist:
            result.append(type_(arg, storage=self.storage).GetScope(type_, arg))

        return result


ScopeType = TypeVar('ScopeType')
T = TypeVar('T')
U = TypeVar('U')


class IScope(Generic[T, U]):

    Applicability: dict[ScopeType, Callable[[ScopeType], bool]]
    Identity: T
    _storage: U

    def __init__(self, id_, storage, context=None):
        self.Identity = id_
        self._storage = storage
        self._context = context
        self._multiply = 1

    def GetStorage(self):
        return self._storage

    def GetScope(self, type_: ScopeType, id_: T, context: str=None):
        if context is None:
            context = self._context
        scope = type_(id_, self._storage, context)

        if hasattr(scope, 'Applicability') and scope.Applicability:
            # if isinstance(scope, PresentValue):
            #     print('hello')
            for subtype, condition in scope.Applicability.items():
                if condition(scope):
                    return subtype(id_, self._storage, context)

        return scope    # Default

    def GetContext(self):
        return self._context



class AllCfIdentities(IScope): # string represents a DataNode

    @property
    def ids(self):
        return [ImportIdentity(AocType = aocStep.AocType,
                              Novelty = aocStep.Novelty,
                              DataNode = self.Identity
                              ) for aocStep in self.GetStorage().GetAllAocSteps(InputSource.Cashflow)]


class GetIdentities(IScope):

    @property
    def computedIdentities(self) -> list[ImportIdentity]:
        return [ImportIdentity(AocType=aocType, Novelty=Novelties.C,  DataNode=self.Identity)
                for aocType in [AocTypes.EA, AocTypes.AM, AocTypes.EOP]]

    @property
    def allIdentities(self) -> set[ImportIdentity]:
        return set(self.ParsedIdentities + self.computedIdentities + self.SpecialIdentities)


    @property
    def ParsedIdentities(self):
        return []

    @property
    def SpecialIdentities(self):
        return []


    #Set DataNode properties and ProjectionPeriod

    @property
    def Identities(self) -> set[ImportIdentity]:
        result = set()

        for id_ in self.allIdentities:
            kwargs = id_.__dict__.copy()
            kwargs['IsReinsurance'] = self.GetStorage().DataNodeDataBySystemName[id_.DataNode].IsReinsurance
            kwargs['ValuationApproach'] = self.GetStorage().DataNodeDataBySystemName[id_.DataNode].ValuationApproach

            result.add(ImportIdentity(**kwargs))

        return result


class AllCashflowIdentities(GetIdentities):

    @property
    def SpecialIdentities(self):
        return self.GetScope(AllCfIdentities, self.Identity).ids


class GetActualIdentities(GetIdentities):

    @property
    def actualEstimateTypes(self) -> list[str]:
        return self.GetStorage().EstimateTypesByImportFormat[ImportFormats.Actual]

    @property
    def ParsedIdentities(self) -> list[ImportIdentity]:
        return [ImportIdentity.from_iv(iv) for iv in self.GetStorage().GetIfrsVariables(self.Identity) if iv.EstimateType in self.actualEstimateTypes]

    @property
    def SpecialIdentities(self) -> list[ImportIdentity]:
        temp = self.GetScope(AllCfIdentities, self.Identity).ids
        temp2 = [ImportIdentity(
            AocType=aocStep.AocType,
            Novelty=aocStep.Novelty,
            DataNode=self.Identity) for aocStep in self.GetStorage().GetAllAocSteps(InputSource.Opening)]

        return temp + temp2


class GetCashflowIdentities(GetIdentities):

    @property
    def isReinsurance(self) -> bool:
        return self.GetStorage().DataNodeDataBySystemName[self.Identity].IsReinsurance #clean up in the next PR

    @property
    def ParsedIdentities(self) -> list[ImportIdentity]:
        # Debug
        # for v in self.GetStorage().GetRawVariables(self.Identity):
        #     print('Stop')

        return [ImportIdentity.from_rv(v) for v in self.GetStorage().GetRawVariables(self.Identity)]

    @property
    def SpecialIdentities(self) -> list[ImportIdentity]:
        temp = {id_.Novelty for id_ in self.ParsedIdentities if id_.Novelty != Novelties.C}    #.Where(id => ).Select(id => ).ToHashSet()
        temp2 = []
        for n in temp:
            if n == Novelties.N:
                temp3 = [AocTypes.IA, AocTypes.CF] #Add IA, CF, for New Business
            elif self.isReinsurance:
                temp3 = [AocTypes.IA, AocTypes.CF, AocTypes.YCU, AocTypes.CRU, AocTypes.RCU]     #Add IA, CF, YCU, CRU, RCU for in force
            else:
                temp3 = [AocTypes.IA, AocTypes.CF, AocTypes.YCU]    #Add IA, CF, YCU

            temp3 = [ImportIdentity(
                AocType = aocType,
                Novelty = n,
                DataNode = self.Identity) for aocType in temp3]

            temp2.extend(temp3)

        temp2.append(ImportIdentity(
               AocType=AocTypes.CF,     #Add CF for Deferral
               Novelty=Novelties.C,
               DataNode=self.Identity))

        temp2.extend([ImportIdentity(
            AocType=aocStep.AocType,
            Novelty=aocStep.Novelty,
            DataNode=self.Identity
                      ) for aocStep in self.GetStorage().GetAllAocSteps(InputSource.Opening)])

        return temp2


class GetAllIdentities(GetIdentities):

    @property
    def SpecialIdentities(self) -> list[ImportIdentity]:
        temp = self.GetScope(AllCfIdentities, self.Identity).ids
        temp2 = [ImportIdentity(AocType = aocStep.AocType,
                                 Novelty = aocStep.Novelty,
                                 DataNode = self.Identity)
                 for aocStep in self.GetStorage().GetAllAocSteps(InputSource.Actual)]

        return temp + temp2


GetIdentities.Applicability = {
    AllCashflowIdentities: lambda x: x.GetStorage().IsSecondaryScope(x.Identity),
    GetActualIdentities: lambda x: x.GetStorage().ImportFormat == ImportFormats.Actual,
    GetCashflowIdentities: lambda x: x.GetStorage().ImportFormat == ImportFormats.Cashflow,
    GetAllIdentities: lambda x: x.GetStorage().ImportFormat == ImportFormats.Opening}


## Getting Amount Types


class ValidAmountType(IScope):  # IScope<string, ImportStorage>
    
    @property
    def BeAmountTypes(self) -> set[str]:
        temp = {rv.AmountType for rv in self.GetStorage().GetRawVariables(self.Identity) if rv.AmountType}
        if self.GetStorage().DataNodeDataBySystemName[self.Identity].IsReinsurance:
            temp.add(AmountTypes.CDR)
        return temp

    @property
    def ActualAmountTypes(self) -> set[str]:
        return {iv.AmountType for iv in self.GetStorage().GetIfrsVariables(self.Identity)
                if iv.EstimateType in self.GetStorage().EstimateTypesByImportFormat[ImportFormats.Actual]}


class BeAmountTypesFromIfrsVariables(ValidAmountType):

    @property
    def BeAmountTypes(self) -> set[str]:
        return {iv.AmountType for iv in self.GetStorage().GetIfrsVariables(self.Identity)
                if iv.EstimateType in self.GetStorage().EstimateTypesByImportFormat[ImportFormats.Cashflow] and iv.AmountType != ''}


ValidAmountType.Applicability = {
    BeAmountTypesFromIfrsVariables: lambda x: x.GetStorage().ImportFormat != ImportFormats.Cashflow or x.GetStorage().IsSecondaryScope(x.Identity)
}

IdentityTuple2 = namedtuple('IdentityTuple2', ['Id', 'AmountType'])

class ParentAocStep(IScope):     #: IScope<(ImportIdentity Id, string AmountType), ImportStorage>

    @property
    def ParsedAocSteps(self) -> set[AocStep]:
        # Debug
        # for id_ in self.GetScope(GetIdentities, self.Identity.Id.DataNode).ParsedIdentities:
        #     print('Stop')

        return {AocStep(id_.AocType, id_.Novelty) for id_ in self.GetScope(GetIdentities, self.Identity.Id.DataNode).ParsedIdentities}

    @property
    def OrderedParsedAocSteps(self) -> list[AocStep]:
        temp = list(self.ParsedAocSteps | set(self.CalculatedTelescopicAocStep))
        return sorted(temp, key=lambda x: self.GetStorage().AocConfigurationByAocStep[x].Order)

    @property
    def ParentParsedIdentities(self) -> dict[AocStep, list[AocStep]]:
        return GetPreviousIdentities(self.OrderedParsedAocSteps)

    @property
    def identityAocStep(self) -> AocStep:
        return AocStep(self.Identity.Id.AocType, self.Identity.Id.Novelty)

    @property
    def CalculatedTelescopicAocStep(self) -> list[AocStep]:
        return self.GetStorage().GetCalculatedTelescopicAocSteps()

    @property
    def Values(self) -> list[AocStep]:

        key = self.Identity.Id.AocType

        if key == AocTypes.CRU:
            return [AocStep(AocTypes.YCU, Novelties.I)]
        elif key == AocTypes.YCU:
            return [GetReferenceAocStepForCalculated(self.OrderedParsedAocSteps, self.GetStorage().AocConfigurationByAocStep, self.identityAocStep)]
        else:
            if parents := self.ParentParsedIdentities.get(self.identityAocStep, None):
                return parents
            else:
                return []


class ParentAocStepForCreditRisk(ParentAocStep):

    @property
    def CalculatedTelescopicAocStep(self) -> list[AocStep]:
        return [aoc for aoc in self.GetStorage().GetCalculatedTelescopicAocSteps() if aoc.AocType != AocTypes.CRU]  #.Where(aoc => )


ParentAocStep.Applicability = {
    ParentAocStepForCreditRisk: lambda x: x.Identity.AmountType != AmountTypes.CDR
}


class ReferenceAocStep(IScope):  #IScope<ImportIdentity, ImportStorage>

    @property
    def OrderedParsedAocSteps(self) -> list[AocStep]:
        temp = {AocStep(id_.AocType, id_.Novelty) for id_ in self.GetScope(GetIdentities, self.Identity.DataNode).ParsedIdentities}
        return sorted(list(temp), key=lambda aocStep: self.GetStorage().AocConfigurationByAocStep[aocStep].Order)

    @property
    def identityAocStep(self) -> AocStep:
        return AocStep(self.Identity.AocType, self.Identity.Novelty)

    def GetReferenceAocStep(self, aocType:str) -> AocStep:

        if aocType in (AocTypes.RCU, AocTypes.CF, AocTypes.IA, AocTypes.YCU, AocTypes.CRU):
            return GetReferenceAocStepForCalculated(self.OrderedParsedAocSteps, self.GetStorage().AocConfigurationByAocStep, self.identityAocStep)

        elif aocType == AocTypes.EA:
            return AocStep(AocTypes.CF, self.Identity.Novelty)

        elif aocType in (AocTypes.AM, AocTypes.EOP):
            return AocStep(AocTypes.CL, Novelties.C)

        elif aocType == AocTypes.BOP:
            return AocStep("", "")  #BOP, C has DataType == Calculated. See ReferenceAocStep condition.

        else:
            raise NotSupportedAocStepReference

    # The Reference AocStep from which get data (Nominal or PV) to compute

    @property
    def Value(self) -> AocStep:
        if (self.GetStorage().AocConfigurationByAocStep[self.identityAocStep].DataType == DataType.Calculated
                     or self.GetStorage().AocConfigurationByAocStep[self.identityAocStep].DataType == DataType.CalculatedTelescopic):
            return self.GetReferenceAocStep(self.Identity.AocType)
        else:
            return self.identityAocStep


IdentityTuple3 = namedtuple('IdentityTuple3', ['Id', 'ScopeInputSource'])


class PreviousAocSteps(IScope):     #<(ImportIdentity Id, InputSource ScopeInputSource), ImportStorage>

    @property
    def  identityAocStep(self) -> AocStep:
        return AocStep(self.Identity.Id.AocType, self.Identity.Id.Novelty)

    @property
    def aocStepOrder(self) -> int:
        return self.GetStorage().AocConfigurationByAocStep[self.identityAocStep].Order

    @property
    def allAocSteps(self) -> {AocStep}:
        return self.GetStorage().GetAllAocSteps(self.Identity.ScopeInputSource)     #.ToHashSet()

    @property
    def Values(self) -> list[AocStep]:

        if self.identityAocStep in self.allAocSteps:

            # ids = {aoc:= AocStep(id_.AocType, id_.Novelty) for id_ in self.GetScope(GetIdentities, self.Identity.Id.DataNode).Identities
            #        if aoc in self.allAocSteps and self.GetStorage().AocConfigurationByAocStep[aoc].Order < self.aocStepOrder
            #        and (aoc.Novelty == self.Identity.Id.Novelty if self.Identity.Id.Novelty != Novelties.C else True)}

            ids = set()
            for id_ in self.GetScope(GetIdentities, self.Identity.Id.DataNode).Identities:
                aoc = AocStep(id_.AocType, id_.Novelty)
                if aoc in self.allAocSteps and self.GetStorage().AocConfigurationByAocStep[aoc].Order < self.aocStepOrder and (
                        aoc.Novelty == self.Identity.Id.Novelty if self.Identity.Id.Novelty != Novelties.C else True):
                    ids.add(aoc)

            return sorted(list(ids), key=lambda aoc: self.GetStorage().AocConfigurationByAocStep[aoc].Order)
        else:
            return []


class MonthlyRate(IScope):
    
    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()    

    @property
    def YearlyYieldCurve(self) -> list[float]:
        return self.GetStorage().GetYearlyYieldCurve(self.Identity, self.EconomicBasis)    

    @property
    def Perturbation(self) -> float:
        return 0 #GetStorage().GetYieldCurvePerturbation() => switch Args.Scenario { 10ptsU => 0.1, 10ptsD => -0.1, _ => default)

    
    @property
    def Interest(self) -> list[float]:
        return [(1 + rate)**(1 / 12) + self.Perturbation for rate in self.YearlyYieldCurve]

                        
    @property
    def Discount(self) -> list[float]:
        return [x ** (-1) for x in self.Interest]


IdentityTuple = namedtuple('IdentityTuple', ['Id', 'AmountType', 'EstimateType', 'AccidentYear'])


class NominalCashflow(IScope):  # <(ImportIdentity Id, string AmountType, string EstimateType, int? AccidentYear), ImportStorage>

    @property
    def referenceAocStep(self) -> AocStep:
        return self.GetScope(ReferenceAocStep, self.Identity.Id).Value

    @property
    def Values(self) -> list[float]:

        importid = dataclasses.replace(self.Identity.Id)
        importid.AocType = self.referenceAocStep.AocType
        importid.Novelty = self.referenceAocStep.Novelty

        return self.GetStorage().GetValues2(importid, self.Identity.AmountType, self.Identity.EstimateType, self.Identity.AccidentYear)


class CreditDefaultRiskNominalCashflow(NominalCashflow):

    Applicability = None

    @property
    def NominalClaimsCashflow(self) -> list[float]:

        claims = self.GetStorage().GetClaims()
        temp = []
        for c in claims:
            importid = dataclasses.replace(self.Identity.Id)
            importid.AocType = self.referenceAocStep.AocType
            importid.Novelty = self.referenceAocStep.Novelty
            temp.append(self.GetStorage().GetValues2(importid, c, self.Identity.EstimateType, self.Identity.AccidentYear))

        return AggregateDoubleArray(temp)
                            
    @property
    def nonPerformanceRiskRate(self) -> float:
        return self.GetStorage().GetNonPerformanceRiskRate(self.Identity.Id)


    @property
    def PvCdrDecumulated(self) -> list[float]:
    
        ret = [0] * len(self.NominalClaimsCashflow)     #new double[NominalClaimsCashflow.Length]
        for i in range(len(self.NominalClaimsCashflow) - 1, -1, -1):
            ret[i] = math.exp(-self.nonPerformanceRiskRate) * (ret[i + 1] if i+1 < len(ret) else 0) + self.NominalClaimsCashflow[i] - (self.NominalClaimsCashflow[i + 1] if i+1 < len(self.NominalClaimsCashflow) else 0)
        return ret

    @property
    def Values(self) -> list[float]:
        return [x - y for x, y in zip(self.PvCdrDecumulated, self.NominalClaimsCashflow)]


class AllClaimsCashflow(NominalCashflow):

    Applicability = None

    @property
    def Values(self) -> list[float]:

        claims = self.GetStorage().GetClaims()
        temp = []
        for c in claims:
            importid = dataclasses.replace(self.Identity.Id)
            importid.AocType = self.referenceAocStep.AocType
            importid.Novelty = self.referenceAocStep.Novelty
            temp.append(self.GetStorage().GetValues2(importid, c, self.Identity.EstimateType, self.Identity.AccidentYear))

        return AggregateDoubleArray(temp)


NominalCashflow.Applicability = {
    CreditDefaultRiskNominalCashflow: lambda x: x.Identity.AmountType == AmountTypes.CDR and x.Identity.Id.AocType == AocTypes.CF,
    AllClaimsCashflow: lambda x: x.Identity.AmountType == AmountTypes.CDR
}

# Discount Cashflow


class DiscountedCashflow(IScope):   #<(ImportIdentity Id, string AmountType, string EstimateType, int? Accidentyear), ImportStorage>

    @property
    def periodType(self) -> PeriodType:
        return self.GetStorage().GetPeriodType(self.Identity.AmountType, self.Identity.EstimateType)

    # static ApplicabilityBuilder ScopeApplicabilityBuilder(ApplicabilityBuilder builder) =>
    #     builder.ForScope<DiscountedCashflow>(s => s.WithApplicability<DiscountedCreditRiskCashflow>(x => x.Identity.Id.IsReinsurance && x.Identity.AmountType == AmountTypes.CDR))

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()

    @property
    def MonthlyDiscounting(self) -> list[float]:
        return self.GetScope(MonthlyRate, self.Identity.Id).Discount

    @property
    def NominalValues(self) -> list[float]:
        return self.GetScope(NominalCashflow, self.Identity).Values

    @property
    def Values(self) -> list[float]:
        return [-1 * x for x in ComputeDiscountAndCumulate(self.NominalValues, self.MonthlyDiscounting, self.periodType)]   # we need to flip the sign to create a reserve view



class DiscountedCreditRiskCashflow(DiscountedCashflow):

    @property
    def nonPerformanceRiskRate(self) -> float:
        return self.GetStorage().GetNonPerformanceRiskRate(self.Identity.Id)


    @property
    def Values(self) -> list[float]:
        return [-1 * x for x in ComputeDiscountAndCumulateWithCreditDefaultRisk(self.NominalValues, self.MonthlyDiscounting, self.nonPerformanceRiskRate)]     # we need to flip the sign to create a reserve view


DiscountedCashflow.Applicability = {
    DiscountedCreditRiskCashflow: lambda x: x.Identity.Id.IsReinsurance and x.Identity.AmountType == AmountTypes.CDR
}



class TelescopicDifference(IScope):      #<(ImportIdentity Id, string AmountType, string EstimateType, int? Accidentyear), ImportStorage>

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()

    @property
    def CurrentValues(self) -> list[float]:
        return self.GetScope(DiscountedCashflow, self.Identity).Values

    @property
    def PreviousValues(self) -> list[float]:
        temp = self.GetScope(ParentAocStep, IdentityTuple2(self.Identity.Id, self.Identity.AmountType)).Values
        temp2 = []
        for aoc in temp:

            id_ = dataclasses.replace(self.Identity.Id)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            temp2.append(self.GetScope(DiscountedCashflow, IdentityTuple(id_, self.Identity.AmountType, self.Identity.EstimateType, self.Identity.AccidentYear)).Values)

        temp2 = [cf for cf in temp2 if len(cf) > 0]

        return AggregateDoubleArray(temp2)

    @property
    def Values(self) -> list[float]:
        return [x - y for x, y in zip(self.CurrentValues, self.PreviousValues)]


class IWithInterestAccretion(IScope):

    @property
    def parentDiscountedValues(self) -> list[float]:
        return [-1 * x for x in self.GetScope(DiscountedCashflow, self.Identity).Values]

    @property
    def parentNominalValues(self) -> list[float]:
        return self.GetScope(NominalCashflow, self.Identity).Values

    @property
    def monthlyInterestFactor(self) -> list[float]:
        return self.GetScope(MonthlyRate, self.Identity.Id).Interest

    def GetInterestAccretion(self) -> list[float]:

        periodType = self.GetStorage().GetPeriodType(self.Identity.AmountType, self.Identity.EstimateType)
        ret = [0] * len(self.parentDiscountedValues)

        if periodType == PeriodType.BeginningOfPeriod:

            for i in range(len(self.parentDiscountedValues)):

                ret[i] = -1 * (self.parentDiscountedValues[i] - self.parentNominalValues[i]) * (
                            GetElementOrDefault(self.monthlyInterestFactor, int(i / 12)) - 1)
        else:
            for i in range(len(self.parentDiscountedValues)):
                ret[i] = -1 * self.parentDiscountedValues[i] * (GetElementOrDefault(self.monthlyInterestFactor, int(i / 12)) - 1)

        return ret


class IWithInterestAccretionForCreditRisk(IScope):

    @property
    def nominalClaimsCashflow(self) -> list[float]:
        return self.GetScope(AllClaimsCashflow, self.Identity).Values

    @property
    def nominalValuesCreditRisk(self) -> list[float]:
        importid = dataclasses.replace(self.Identity.Id)
        importid.AocType = AocTypes.CF
        kwargs = self.Identity._asdict()
        kwargs['Id'] = importid
        identity = IdentityTuple(**kwargs)

        return -1 * self.GetScope(CreditDefaultRiskNominalCashflow, identity).Values

    @property
    def monthlyInterestFactor(self) -> list[float]:
        return self.GetScope(MonthlyRate, self.Identity.Id).Interest

    @property
    def nonPerformanceRiskRate(self) -> float:
        return self.GetStorage().GetNonPerformanceRiskRate(self.Identity.Id)

    def GetInterestAccretion(self) -> list[float]:

        interestOnClaimsCashflow =  [0] * len(self.nominalClaimsCashflow)
        interestOnClaimsCashflowCreditRisk = [0] * len(self.nominalClaimsCashflow)
        effectCreditRisk = [0] * len(self.nominalClaimsCashflow)

        for i in range(len(self.nominalClaimsCashflow) - 1, -1, -1):        #(var i = nominalClaimsCashflow.Length - 1; i >= 0; i--)

            interestOnClaimsCashflow[i] = 1 / GetElementOrDefault(self.monthlyInterestFactor, int(i/12)) * (
                    (interestOnClaimsCashflow[i + 1] if i+1 < len(interestOnClaimsCashflow) else 0) + self.nominalClaimsCashflow[i] - (self.nominalClaimsCashflow[i + 1] if i+1 < len(self.nominalClaimsCashflow) else 0))
            interestOnClaimsCashflowCreditRisk[i] = 1 / GetElementOrDefault(self.monthlyInterestFactor, int(i/12)) * (
                    math.exp(-self.nonPerformanceRiskRate) * (interestOnClaimsCashflowCreditRisk[i + 1] if i+1 < len(interestOnClaimsCashflowCreditRisk) else 0) + self.nominalClaimsCashflow[i] - (self.nominalClaimsCashflow[i + 1] if i+1 < len(self.nominalClaimsCashflow) else 0))
            effectCreditRisk[i] = interestOnClaimsCashflow[i] - interestOnClaimsCashflowCreditRisk[i]

        return [x - y for x, y in zip(self.nominalValuesCreditRisk, effectCreditRisk)]


class IWithGetValueFromValues(IScope):      # IScope<(ImportIdentity Id, string AmountType, string EstimateType, int? AccidentYear), ImportStorage>{

    @property
    def shift(self) -> int:
        return self.GetStorage().GetShift(0)  #Identity.Id.ProjectionPeriod

    @property
    def timeStep(self) -> int:
        return self.GetStorage().GetTimeStep(0)    #Identity.Id.ProjectionPeriod

    def GetValueFromValues(self, Values: list[float]) -> float:

        key = self.GetStorage().GetValuationPeriod(self.Identity.Id)

        if key == ValuationPeriod.BeginningOfPeriod:
            return Values[self.shift] if self.shift < len(Values) else 0.0

        elif key == ValuationPeriod.MidOfPeriod:
            idx = self.shift + round(self.timeStep / 2) - 1
            return Values[idx] if idx < len(Values) else 0.0

        elif key == ValuationPeriod.Delta:
            return sum(Values[self.shift:][:self.timeStep])

        elif key == ValuationPeriod.EndOfPeriod:
            return Values[self.shift + self.timeStep] if self.shift + self.timeStep < len(Values) else 0

        elif key == ValuationPeriod.NotApplicable:
            return 0

        else:
            raise RuntimeError('must not happen')


class PresentValue(IWithGetValueFromValues): 

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()

    @property
    def Values(self) -> list[float]:
        return self.GetScope(TelescopicDifference, self.Identity).Values

    @property
    def Value(self) -> float:
        return self._multiply * self.GetValueFromValues(self.Values)


class ComputePresentValueWithIfrsVariable(PresentValue):


    @property
    def Value(self) -> list[float]:
        return self._multiply * self.GetStorage().GetValue(
            self.Identity.Id, self.Identity.AmountType, self.Identity.EstimateType, economicBasis=EconomicBasis, accidentYear=self.Identity.AccidentYear)

    @property
    def Values(self) -> list[float]:
        return []


class PresentValueFromDiscountedCashflow(PresentValue):

    @property
    def Values(self) -> list[float]:
        return self.GetScope(DiscountedCashflow, self.Identity).Values


class CashflowAocStep(PresentValue):

    @property
    def Values(self) -> list[float]:
        return self.GetScope(NominalCashflow, self.Identity).Values


class PresentValueWithInterestAccretion(PresentValue, IWithInterestAccretion):

    @property
    def Values(self) -> list[float]:
        return self.GetInterestAccretion()


class PresentValueWithInterestAccretionForCreditRisk(PresentValue, IWithInterestAccretionForCreditRisk):

    @property
    def Values(self) -> list[float]:
        return self.GetInterestAccretion()


class EmptyValuesAocStep(PresentValue):

    @property
    def Values(self) -> list[float]:
        return []


PresentValue.Applicability = {
            ComputePresentValueWithIfrsVariable: lambda x: x.GetStorage().ImportFormat != ImportFormats.Cashflow or x.GetStorage().IsSecondaryScope(x.Identity.Id.DataNode),
            PresentValueFromDiscountedCashflow: lambda x: (x.Identity.Id.AocType == AocTypes.BOP and x.Identity.Id.Novelty != Novelties.C) or x.Identity.Id.AocType == AocTypes.EOP,
            CashflowAocStep: lambda x: x.Identity.Id.AocType == AocTypes.CF,
            PresentValueWithInterestAccretionForCreditRisk: lambda x: x.Identity.Id.IsReinsurance and x.Identity.AmountType == AmountTypes.CDR and x.Identity.Id.AocType == AocTypes.IA,
            PresentValueWithInterestAccretion: lambda x: x.Identity.Id.AocType == AocTypes.IA,
            EmptyValuesAocStep: lambda x: x.Identity.Id.AocType in [AocTypes.BOP, AocTypes.EA, AocTypes.AM, AocTypes.RCU]   #add here combination CRU for At !CDR?
}


class PvLocked(IScope):     #<ImportIdentity, ImportStorage>

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.L

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.BE

    @property
    def accidentYears(self) -> list[int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)

    @property
    def PresentValues(self) -> list[PresentValue]:
        temp = self.GetScope(ValidAmountType, self.Identity.DataNode).BeAmountTypes
        temp2 = []
        for at in temp:

            # # Debug
            # for ay in self.accidentYears:
            #     temp3 = self.GetScope(PresentValue, IdentityTuple(self.Identity, at, self.EstimateType, ay), self.EconomicBasis)

            temp2 += [self.GetScope(PresentValue, IdentityTuple(self.Identity, at, self.EstimateType, ay), self.EconomicBasis) for ay in self.accidentYears]

        return temp2

    @property
    def Value(self) -> float:
        return sum(self.PresentValues)


class PvCurrent(IScope):    #<ImportIdentity, ImportStorage>

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.C

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.BE

    @property
    def accidentYears(self) -> list[int]:
        return list(self.GetStorage().GetAccidentYears(self.Identity.DataNode))

    @property
    def PresentValues(self) -> list[PresentValue]:
        temp = self.GetScope(ValidAmountType, self.Identity.DataNode).BeAmountTypes
        temp2 = []
        for at in temp:
            temp2 += [self.GetScope(PresentValue, IdentityTuple(self.Identity, at, self.EstimateType, ay), self.EconomicBasis) for ay in self.accidentYears]

        return temp2

    @property
    def Value(self):
        return sum(self.PresentValues)



class RaLocked(IScope):

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.L

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.RA

    @property
    def accidentYears(self) -> [int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)

    @property
    def PresentValues(self) -> [PresentValue]:
        return [self.GetScope(PresentValue, IdentityTuple(self.Identity, None, self.EstimateType, ay), self.EconomicBasis) for ay in self.accidentYears]

    @property
    def Value(self) -> float:
        return sum([pv.Value for pv in self.PresentValues])          # self.PresentValues.Aggregate().Value


class RaCurrent(IScope):

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.C

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.RA

    @property
    def accidentYears(self) -> [int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)

    @property
    def PresentValues(self) -> [PresentValue]:
        return [self.GetScope(PresentValue, IdentityTuple(self.Identity, None, self.EstimateType, ay), self.EconomicBasis) for ay in self.accidentYears]

    @property
    def Value(self) -> float:
        return sum([pv.Value for pv in self.PresentValues])



class PvToIfrsVariable(IScope):

    @property
    def PvLocked(self) -> list[IfrsVariable]:

        # For debug
        # temp = self.GetScope(PvLocked, self.Identity)
        # for temp2 in temp.PresentValues:
        #     temp3 = temp2.Value

        result = []
        for x in [pvs for pvs in self.GetScope(PvLocked, self.Identity).PresentValues if abs(pvs.Value) >= Precision]:

            result.append(IfrsVariable(
                Id=uuid.uuid4(),
                EconomicBasis = x.EconomicBasis,
                EstimateType = x.Identity.EstimateType,
                DataNode = x.Identity.Id.DataNode,
                AocType = x.Identity.Id.AocType,
                Novelty = x.Identity.Id.Novelty,
                AccidentYear = x.Identity.AccidentYear,
                AmountType = x.Identity.AmountType,
                Value = x.Value,
                Partition = self.GetStorage().TargetPartition))

        return result


    @property
    def PvCurrent(self) -> list[IfrsVariable]:

        result = []
        for x in [x for x in self.GetScope(PvCurrent, self.Identity).PresentValues if abs(x.Value) >= Precision]:
            result.append(IfrsVariable(
                Id=uuid.uuid4(),
                EconomicBasis = x.EconomicBasis,
                EstimateType = x.Identity.EstimateType,
                DataNode = x.Identity.Id.DataNode,
                AocType = x.Identity.Id.AocType,
                Novelty = x.Identity.Id.Novelty,
                AccidentYear = x.Identity.AccidentYear,
                AmountType = x.Identity.AmountType,
                Value = x.Value,
                Partition = self.GetStorage().TargetPartition))

        return result


class RaToIfrsVariable(IScope):     # <ImportIdentity, ImportStorage>

    @property
    def RaCurrent(self) -> [IfrsVariable]:

        # # Debug
        # for x in self.GetScope(RaCurrent, self.Identity).PresentValues:
        #     if self.Identity.Novelty == 'N' and self.Identity.AocType == 'BOP':
        #         print(x.Value)

        result = []
        for x in [x for x in self.GetScope(RaCurrent, self.Identity).PresentValues if abs(x.Value) >= Precision]:

            result.append(IfrsVariable(
                Id=uuid.uuid4(),
                EconomicBasis = x.EconomicBasis,
                EstimateType = x.Identity.EstimateType,
                DataNode = x.Identity.Id.DataNode,
                AocType = x.Identity.Id.AocType,
                Novelty = x.Identity.Id.Novelty,
                AccidentYear = x.Identity.AccidentYear,
                AmountType = '',
                Value = x.Value,
                Partition = self.GetStorage().TargetPartition
                ))
        return result

    @property
    def RaLocked(self) -> [IfrsVariable]:

        result = []
        for x in [x for x in self.GetScope(RaLocked, self.Identity).PresentValues if abs(x.Value) >= Precision]:
            result.append(IfrsVariable(
                Id=uuid.uuid4(),
                EconomicBasis = x.EconomicBasis,
                EstimateType = x.Identity.EstimateType,
                DataNode = x.Identity.Id.DataNode,
                AocType = x.Identity.Id.AocType,
                Novelty = x.Identity.Id.Novelty,
                AccidentYear = x.Identity.AccidentYear,
                AmountType = '',
                Value = x.Value,
                Partition = self.GetStorage().TargetPartition
            ))
        return result


class CoverageUnitCashflow(IScope):      #<ImportIdentity, ImportStorage>

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.CU

    @property
    def Values(self) -> [float]:
        return self.GetScope(DiscountedCashflow, IdentityTuple(self.Identity, '', self.EstimateType, None)).Values


class MonthlyAmortizationFactorCashflow(IScope):     #<ImportIdentity, ImportStorage>

    @property
    def NominalCuCashflow(self) -> [float]:
    
        id_ = dataclasses.replace(self.Identity)
        id_.AocType = AocTypes.CL
        
        return self.GetScope(NominalCashflow, IdentityTuple(id_, '', EstimateTypes.CU, None)).Values

    @property
    def DiscountedCuCashflow(self) -> [float]:
    
        id_ = dataclasses.replace(self.Identity)
        id_.AocType = AocTypes.CL
        return [-1 * x  for x in self.GetScope(CoverageUnitCashflow, id_, self.EconomicBasis).Values]

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()
    
    @property
    def MonthlyAmortizationFactors(self) -> [float]:

        if self.Identity.AocType == AocTypes.AM:

            result = []
            for nominal, discountedCumulated in zip(self.NominalCuCashflow, self.DiscountedCuCashflow):
                if abs(discountedCumulated) >= Precision:
                    result.append(1 - nominal / discountedCumulated)
                else:
                    result.append(0)

            return result
        else:
            return []


class CurrentPeriodAmortizationFactor(IScope):  #<ImportIdentity, ImportStorage>

    @property
    def shift(self) -> int:
        return self.GetStorage().GetShift(0)           # Identity.ProjectionPeriod

    @property
    def timeStep(self) -> int:
        return self.GetStorage().GetTimeStep(0)     # Identity.ProjectionPeriod

    @property
    def amortizedFactor(self) -> float:
        temp = self.GetScope(MonthlyAmortizationFactorCashflow, self.Identity).MonthlyAmortizationFactors
        return math.prod(temp[self.shift: self.shift + self.timeStep])

    @property
    def EconomicBasis(self) -> str:
        return self.GetContext()

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.F

    @property
    def Value(self) -> float:
        return 1 - self.amortizedFactor if abs(self.amortizedFactor - 1) > Precision else 1.0


class AmfFromIfrsVariable(CurrentPeriodAmortizationFactor):

    @property
    def Value(self) -> float:
        return self.GetStorage().GetValue(self.Identity, '', self.EstimateType, economicBasis=self.EconomicBasis, accidentYear=None)


CurrentPeriodAmortizationFactor.Applicability = {
    AmfFromIfrsVariable: lambda x: x.GetStorage().ImportFormat != ImportFormats.Cashflow or x.GetStorage().IsSecondaryScope(x.Identity.DataNode)
}


class ActualBase(IScope):    # <(ImportIdentity Id, string AmountType, string EstimateType, int? AccidentYear), ImportStorage>


        # static ApplicabilityBuilder ScopeApplicabilityBuilder(ApplicabilityBuilder builder) =>
        #
        #         builder.ForScope<ActualBase>(s => s.WithApplicability<EmptyValuesActual>(x => x.GetStorage().ImportFormat == ImportFormats.Actual
        #                                                                                    && !x.GetStorage().IsSecondaryScope(x.Identity.Id.DataNode)
        #                                                                                    && x.Identity.Id.AocType == AocTypes.AM)
        #                                            .WithApplicability<EndOfPeriodActual>(x => x.GetStorage().ImportFormat != ImportFormats.Cashflow
        #                                                                                    && !x.GetStorage().IsSecondaryScope(x.Identity.Id.DataNode)
        #                                                                                    && x.Identity.Id.AocType == AocTypes.EOP
        #                                                                                    && x.Identity.EstimateType != EstimateTypes.A)
        #
        #                                );

    @property
    def Value(self) -> float:
        return self._multiply * self.GetStorage().GetValue(
            self.Identity.Id, self.Identity.AmountType,
            economicBasis=self.Identity.EstimateType,
            accidentYear=self.Identity.AccidentYear)


class EndOfPeriodActual(ActualBase):

    @property
    def Value(self) -> float:

        result = []

        for aocStep in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity.Id, InputSource.Actual)).Values:
            id_ = dataclasses.replace(self.Identity.Id)
            id_.AocType = aocStep.AocType
            id_.Novelty = aocStep.Novelty

            result.append(self.GetScope(ActualBase,
                                        IdentityTuple(id_, self.Identity.AmountType, self.Identity.EstimateType, self.Identity.AccidentYear)).Value)

        return self._multiply * sum(result)


class EmptyValuesActual(ActualBase):

    @property
    def Value(self):
        return 0


ActualBase.Applicability = {
    EmptyValuesActual: lambda x: (x.GetStorage().ImportFormat == ImportFormats.Actual
                                  and not x.GetStorage().IsSecondaryScope(x.Identity.Id.DataNode)
                                  and x.Identity.Id.AocType == AocTypes.AM),
    EndOfPeriodActual: lambda x: (x.GetStorage().ImportFormat != ImportFormats.Cashflow
                                  and not x.GetStorage().IsSecondaryScope(x.Identity.Id.DataNode)
                                  and x.Identity.Id.AocType == AocTypes.EOP
                                  and x.Identity.EstimateType != EstimateTypes.A)
}


class Actual(IScope):     #<ImportIdentity, ImportStorage>

    # [IdentityProperty][NotVisible][Dimension(typeof(EstimateType))]

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.A

    @property
    def accidentYears(self) -> [int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)  #.ToArray()

    # [NotVisible]

    @property
    def Actuals(self) -> [ActualBase]:
        result = []
        for at_ in self.GetScope(ValidAmountType, self.Identity.DataNode).ActualAmountTypes:
            result.extend(
                [self.GetScope(ActualBase, IdentityTuple(self.Identity, at_, EstimateType, ay)) for ay in self.accidentYears]
            )
        return result


class AdvanceActual(IScope):     #<ImportIdentity, ImportStorage>

    # [IdentityProperty][NotVisible][Dimension(typeof(EstimateType))]

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.AA

    @property
    def accidentYears(self) -> [int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)    #.ToArray()

    @property
    def Actuals(self) -> [ActualBase]:
        result = []
        for at_ in self.GetScope(ValidAmountType, self.Identity.DataNode).ActualAmountTypes:
            result.extend(
                [self.GetScope(ActualBase, IdentityTuple(self.Identity, at_, EstimateType, ay)) for ay in self.accidentYears]
            )
        return result


class OverdueActual(IScope):    #<ImportIdentity, ImportStorage>

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.OA;


    @property
    def accidentYears(self) -> [int]:
        return self.GetStorage().GetAccidentYears(self.Identity.DataNode)


    @property
    def Actuals(self) -> [ActualBase]:
        result = []
        for at_ in self.GetScope(ValidAmountType, self.Identity.DataNode).ActualAmountTypes:
            result.extend(
                [self.GetScope(ActualBase, IdentityTuple(self.Identity, at_, EstimateType, ay)) for ay in self.accidentYears]
            )
        return result



class DeferrableActual(IScope):     #<ImportIdentity, ImportStorage>

    # static ApplicabilityBuilder ScopeApplicabilityBuilder(ApplicabilityBuilder builder) =>
    #
    #         builder.ForScope<DeferrableActual>(s => s.WithApplicability<>()
    #                                                  .WithApplicability<>(x => x.Identity.AocType == AocTypes.CF)
    #                                                  .WithApplicability<>(x => x.Identity.AocType == AocTypes.AM)
    #                                                  .WithApplicability<>(x => x.Identity.AocType == AocTypes.EOP)

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.DA
    
    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.L
        
    @property
    def Value(self) -> float:
        return self.GetStorage().GetValue(self.Identity, '', EstimateType)  #, None)


class DeferrableActualForCurrentBasis(DeferrableActual):

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.C


class ReleaseDeferrable(DeferrableActual):

    @property
    def Value(self) -> float:
        return sum([self.GetScope(ActualBase, IdentityTuple(self.Identity, at_, EstimateTypes.A, None)).Value
                    for at_ in self.GetStorage().GetAttributableExpenseAndCommissionAmountType()])


class AmortizationDeferrable(DeferrableActual):

    @property
    def AmortizationFactor(self) -> float:
        return self.GetScope(CurrentPeriodAmortizationFactor, self.Identity, self.EconomicBasis).Value


    @property
    def AggregatedDeferrable(self) -> float:

        result = []
        for aocStep in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Actual)).Values:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aocStep.AocType
            id_.Novelty = aocStep.Novelty
            result.append(self.GetScope(DeferrableActual, id_).Value)

        return sum(result)

    @property
    def Value(self) -> float:
        return -1 * self.AggregatedDeferrable * self.AmortizationFactor


class EndOfPeriodDeferrable(DeferrableActual):

    @property
    def Value(self) -> float:

        result = []
        for aocStep in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Actual)).Values:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aocStep.AocType
            id_.Novelty = aocStep.Novelty
            result.append(self.GetScope(DeferrableActual, id_).Value)

        return sum(result)


"""
x => x.Identity.ValuationApproach == ValuationApproaches.VFA, p => p.ForMember(s => s.EconomicBasis)
"""


DeferrableActual.Applicability = {
    DeferrableActualForCurrentBasis: lambda x: x.Identity.ValuationApproach == ValuationApproaches.VFA,     # p => p.ForMember(s => s.EconomicBasis)
    ReleaseDeferrable: lambda x: x.Identity.AocType == AocTypes.CF,
    AmortizationDeferrable: lambda x: x.Identity.AocType == AocTypes.AM,
    EndOfPeriodDeferrable: lambda x: x.Identity.AocType == AocTypes.EOP
}


class BeExperienceAdjustmentForPremium(IScope):     # <ImportIdentity, ImportStorage>

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.BEPA

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.L

    @property
    def ByAmountType(self) -> [PresentValue]:
        mlt = self.GetStorage().GetPremiumAllocationFactor(self.Identity)
        result = []
        for pr in self.GetStorage().GetPremiums():
            pv = self.GetScope(PresentValue, IdentityTuple(self.Identity, pr, EstimateTypes.BE, None), self.EconomicBasis)
            pv._multiply = mlt
            result.append(pv)

        return result


class DefaultValueBeExperienceAdjustmentForPremium(BeExperienceAdjustmentForPremium):

    @property
    def ByAmountType(self) -> [PresentValue]:
        return []


BeExperienceAdjustmentForPremium.Applicability = {
    DefaultValueBeExperienceAdjustmentForPremium: lambda x: x.Identity.AocType != AocTypes.CF
}


class ActualExperienceAdjustmentOnPremium(IScope):   #<ImportIdentity, ImportStorage>


    @property
    def ByAmountTypeAndEstimateType(self) -> [ActualBase]:
        temp = self.GetStorage().GetPremiums()
        mlt = self.GetStorage().GetPremiumAllocationFactor(self.Identity)
        result = []
        for pr in temp:
            pv = self.GetScope(ActualBase, IdentityTuple(self.Identity, pr, EstimateTypes.A, None))
            pv._multiply = mlt
            result.append(pv)
        return result


class DefaultValueActualExperienceAdjustmentOnPremium(ActualExperienceAdjustmentOnPremium):

    @property
    def ByAmountTypeAndEstimateType(self) -> [ActualBase]:
        return []


ActualExperienceAdjustmentOnPremium.Applicability = {
    DefaultValueActualExperienceAdjustmentOnPremium: lambda x: x.Identity.AocType != AocTypes.CF
}


class TechnicalMargin(IScope):  #<ImportIdentity, ImportStorage>


    # static ApplicabilityBuilder ScopeApplicabilityBuilder(ApplicabilityBuilder builder) =>
    #     builder.ForScope<TechnicalMargin>(s => s.WithApplicability<TechnicalMarginForCurrentBasis>(x => x.Identity.ValuationApproach == ValuationApproaches.VFA, p => p.ForMember(s => s.EconomicBasis))
    #                                            .WithApplicability<TechnicalMarginForBOP>(x => x.Identity.AocType == AocTypes.BOP && x.Identity.Novelty == Novelties.I)
    #                                            .WithApplicability<TechnicalMarginDefaultValue>(x => x.Identity.AocType == AocTypes.CF)
    #                                            .WithApplicability<TechnicalMarginForIA>(x => x.Identity.AocType == AocTypes.IA && x.Identity.Novelty == Novelties.I)
    #                                            .WithApplicability<TechnicalMarginForEA>(x => x.Identity.AocType == AocTypes.EA && !x.Identity.IsReinsurance)
    #                                            .WithApplicability<TechnicalMarginForAM>(x => x.Identity.AocType == AocTypes.AM)
    #                                            )
    
    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.L
    
    @property
    def Value(self) -> float:

        x = self.GetScope(ValidAmountType, self.Identity.DataNode).BeAmountTypes
        y = self.GetStorage().GetNonAttributableAmountType()
        z = x - y

        temp1 = sum([self.GetScope(PresentValue, IdentityTuple(self.Identity, at_, EstimateTypes.BE, None), self.EconomicBasis).Value for at_ in z])
        temp2 = self.GetScope(RaLocked, self.Identity).Value
        return temp1 + temp2

    @property
    def AggregatedValue(self) -> float:
        
        result = []
        for aoc in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Cashflow)).Values:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            result.append(self.GetScope(TechnicalMargin, id_).Value)
        
        return sum(result)


class TechnicalMarginForCurrentBasis(TechnicalMargin): 

    @property
    def EconomicBasis(self):
        return EconomicBases.C


class TechnicalMarginForBOP(TechnicalMargin): 

    @property
    def ValueCsm(self) -> float:
        return self.GetStorage().GetValue(self.Identity, '', estimateType=EstimateTypes.C, accidentYear=None)
    
    @property
    def ValueLc(self) -> float:
        return self.GetStorage().GetValue(self.Identity, '', estimateType=EstimateTypes.L, accidentYear=None)
    
    @property
    def ValueLr(self) -> float:
        return self.GetStorage().GetValue(self.Identity, '', estimateType=EstimateTypes.LR, accidentYear=None)
    
    @property
    def Value(self) -> float:
        return -1 * self.ValueCsm + self.ValueLc + self.ValueLr


class TechnicalMarginDefaultValue(TechnicalMargin): 

    @property
    def Value(self):
        return 0    #=> default


class TechnicalMarginForIA(TechnicalMargin):

    @property
    def timeStep(self) -> int:
        return self.GetStorage().GetTimeStep(0)             #Identity.Id.ProjectionPeriod

    @property
    def shift(self) -> int:
        return self.GetStorage().GetShift(0)          #Identity.Id.ProjectionPeriod

    
    @property
    def monthlyInterestFactor(self) -> [float]:
        return self.GetScope(MonthlyRate, self.Identity, self.EconomicBasis).Interest
    
    @property
    def interestAccretionFactor(self) -> float:
        result = []
        for i in range(self.shift, self.timeStep):
            result.append(self.monthlyInterestFactor[int(i/12)] if len(self.monthlyInterestFactor) < int(i/12) else 0)

        return math.prod(result) - 1


    @property
    def Value(self) -> float:
        return self.AggregatedValue * self.interestAccretionFactor


class TechnicalMarginForEA(TechnicalMargin):

    # static ApplicabilityBuilder ScopeApplicabilityBuilderInner(ApplicabilityBuilder builder) =>
    #     builder.ForScope<TechnicalMargin>(s => s.WithApplicability<TechnicalMarginDefaultValue>(x => x.Identity.IsReinsurance))
                                               
    @property
    def referenceAocType(self) -> str:
        return self.GetScope(ReferenceAocStep, self.Identity).Value.AocType
    
    @property
    def premiums(self) -> float:

        result = []
        for n in self.GetStorage().GetNovelties3(self.referenceAocType, InputSource.Cashflow):
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = self.referenceAocType
            id_.Novelty = n
            temp = sum([sc.Value for sc in self.GetScope(BeExperienceAdjustmentForPremium, id_).ByAmountType])
            id_.Novelty = Novelties.C
            temp2 = sum([sc.Value for sc in self.GetScope(ActualExperienceAdjustmentOnPremium, id_).ByAmountTypeAndEstimateType])
            result.append(temp - temp2)

        return sum(result)

    @property
    def attributableExpenseAndCommissions(self) -> float:

        result = []
        for d in self.GetStorage().GetAttributableExpenseAndCommissionAmountType():
            temp = self.GetStorage().GetNovelties3(self.referenceAocType, InputSource.Cashflow)
            result2 = []
            for n in temp:
                id_ = dataclasses.replace(self.Identity)
                id_.AocType = self.referenceAocType
                id_.Novelty = n
                temp2 = self.GetScope(PresentValue, IdentityTuple(id_, d, EstimateTypes.BE, None), self.EconomicBasis).Value
                id_.Novelty = Novelties.C
                temp3 = self.GetScope(ActualBase, IdentityTuple(id_, d, EstimateTypes.A, None)).Value
                result2.append(temp2 - temp3)
            result.append(sum(result2))

        return sum(result)


    @property
    def investmentClaims(self) -> float:

        result = []
        for ic in self.GetStorage().GetInvestmentClaims():
            result2 = []
            for n in self.GetStorage().GetNovelties3(self.referenceAocType, InputSource.Cashflow):
                id_ = dataclasses.replace(self.Identity)
                id_.AocType = self.referenceAocType
                id_.Novelty = n
                temp1 = self.GetScope(PresentValue, IdentityTuple(id_, ic, EstimateTypes.BE, None), self.EconomicBasis).Value
                id_.Novelty = Novelties.C
                temp2 = self.GetScope(ActualBase, IdentityTuple(id_, ic, EstimateTypes.A, None)).Value
                result2.append(temp1 - temp2)
            result.append(sum(result2))

        return sum(result)

    @property
    def Value(self) -> float:
        return self.premiums + self.attributableExpenseAndCommissions + self.investmentClaims


class TechnicalMarginForAM(TechnicalMargin):

    @property
    def Value(self):
        return -1 * self.AggregatedValue * self.GetScope(CurrentPeriodAmortizationFactor, self.Identity, self.EconomicBasis).Value


TechnicalMargin.Applicability = {
    TechnicalMarginForCurrentBasis: lambda x: x.Identity.ValuationApproach == ValuationApproaches.VFA,  #  p => p.ForMember(s => s.EconomicBasis)
    TechnicalMarginForBOP: lambda x: x.Identity.AocType == AocTypes.BOP and x.Identity.Novelty == Novelties.I,
    TechnicalMarginDefaultValue: lambda x: x.Identity.AocType == AocTypes.CF,
    TechnicalMarginForIA: lambda x: x.Identity.AocType == AocTypes.IA and x.Identity.Novelty == Novelties.I,
    TechnicalMarginForEA: lambda x: x.Identity.AocType == AocTypes.EA and not x.Identity.IsReinsurance,
    TechnicalMarginForAM: lambda x: x.Identity.AocType == AocTypes.AM
    }


TechnicalMarginForEA.Applicability = {
    TechnicalMarginDefaultValue: lambda x: x.Identity.IsReinsurance
}


class AllocateTechnicalMargin(IScope):  #<ImportIdentity, ImportStorage>


    # Switch

    # static ApplicabilityBuilder ScopeApplicabilityBuilder(ApplicabilityBuilder builder) =>
    #
    #     builder.ForScope<AllocateTechnicalMargin>(s => s
    #                                          .WithApplicability<AllocateTechnicalMarginForReinsuranceCL>(x => x.Identity.IsReinsurance && x.Identity.AocType == AocTypes.CL)
    #                                          .WithApplicability<AllocateTechnicalMarginForReinsurance>(x => x.Identity.IsReinsurance,
    #                                                                                                    p => p.ForMember(s => s.ComputedEstimateType)
    #                                                                                                          .ForMember(s => s.HasSwitch))
    #
    #                                          .WithApplicability<ComputeAllocateTechnicalMarginWithIfrsVariable>(x => x.GetStorage().IsSecondaryScope(x.Identity.DataNode))
    #                                          .WithApplicability<AllocateTechnicalMarginForBop>(x => x.Identity.AocType == AocTypes.BOP)
    #                                          .WithApplicability<AllocateTechnicalMarginForCl>(x => x.Identity.AocType == AocTypes.CL)
    #                                          .WithApplicability<AllocateTechnicalMarginForEop>(x => x.Identity.AocType == AocTypes.EOP)
    #
    #                                          )
    
    @property
    def AggregatedTechnicalMargin(self) -> float:
        return self.GetScope(TechnicalMargin, self.Identity).AggregatedValue

    @property
    def TechnicalMargin(self) -> float:
        return self.GetScope(TechnicalMargin, self.Identity).Value
    
    @property
    def ComputedEstimateType(self) -> str:
        return self.ComputeEstimateType(self.GetScope(TechnicalMargin, self.Identity).AggregatedValue + self.TechnicalMargin)

    @property
    def HasSwitch(self) -> bool:
        return self.ComputedEstimateType != self.ComputeEstimateType(self.GetScope(TechnicalMargin, self.Identity).AggregatedValue)

    # Allocate
    @property
    def EstimateType(self) -> str:
        return self.GetContext()
    
    @property
    def Value(self) -> float:

        if self.HasSwitch and self.EstimateType == self.ComputedEstimateType:
            return self.TechnicalMargin + self.AggregatedTechnicalMargin

        elif self.HasSwitch and not self.EstimateType == self.ComputedEstimateType:
            return -1 * self.AggregatedTechnicalMargin

        elif not self.HasSwitch and self.EstimateType == self.ComputedEstimateType:
            return self.TechnicalMargin

        else:
            return 0

    def ComputeEstimateType(self, aggregatedTechnicalMargin: float) -> str:
        return EstimateTypes.L if aggregatedTechnicalMargin > Precision else EstimateTypes.C



class ComputeAllocateTechnicalMarginWithIfrsVariable(AllocateTechnicalMargin):

    @property
    def TechnicalMargin(self) -> float:
        return self.ComputeTechnicalMarginFromIfrsVariables(self.Identity)

    @property
    def AggregatedTechnicalMargin(self) -> float:
        result = []
        for aoc in self.GetScope(PreviousAocSteps, (self.Identity, InputSource.Cashflow)).Values:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            result.append(self.ComputeTechnicalMarginFromIfrsVariables(id_))

        return sum(result)

    def ComputeTechnicalMarginFromIfrsVariables(self, id_: ImportIdentity):

        return (self.GetStorage().GetValue(self.Identity, '', EstimateTypes.LR, None) +
                 self.GetStorage().GetValue(self.Identity, '', EstimateTypes.L, None) -
               self.GetStorage().GetValue(self.Identity, '', EstimateTypes.C, None))


class AllocateTechnicalMarginForReinsurance(AllocateTechnicalMargin):


   # TODO add Reinsurance Coverage Update (RCU, Novelty=I) AocStep

    @property
    def underlyingGic(self) -> [list]:
        return self.GetStorage().GetUnderlyingGic(self.Identity)
   
    @property
    def weightedUnderlyingTM(self) -> float:

        result = []
        for gic in self.underlyingGic:
            id_ = dataclasses.replace(self.Identity)
            id_.DataNode = gic
            result.append(self.GetStorage().GetReinsuranceCoverage(self.Identity, gic) * self.GetScope(AllocateTechnicalMargin, id_).TechnicalMargin)

        return sum(result)
                                                                      
    @property
    def weightedUnderlyingAggregatedTM(self) -> float:
        result = []
        for gic in self.underlyingGic:
            id_ = dataclasses.replace(self.Identity)
            id_.DataNode = gic
            result.append(self.GetStorage().GetReinsuranceCoverage(self.Identity, gic) * self.GetScope(AllocateTechnicalMargin, id_).AggregatedTechnicalMargin)

        return sum(result)

    def ComputeReinsuranceEstimateType(self, aggregatedFcf: float) -> str:
        return EstimateTypes.LR if aggregatedFcf > Precision else EstimateTypes.C
    
    @property
    def ComputedEstimateType(self) -> str:
        return self.ComputeReinsuranceEstimateType(self.weightedUnderlyingAggregatedTM + self.weightedUnderlyingTM)

    @property
    def HasSwitch(self) -> bool:
        return self.ComputedEstimateType != self.ComputeReinsuranceEstimateType(self.weightedUnderlyingAggregatedTM)



class AllocateTechnicalMarginForReinsuranceCL(AllocateTechnicalMargin):


    # In common1

    @property
    def underlyingGic(self) -> list[str]:
        return self.GetStorage().GetUnderlyingGic(self.Identity)
   
    @property
    def weightedUnderlyingTM(self) -> float:
        result = []
        for gic in self.underlyingGic:
            id_ = dataclasses.replace(self.Identity)
            id_.DataNode = gic
            result.append(self.GetStorage().GetReinsuranceCoverage(self.Identity, gic) * self.GetScope(AllocateTechnicalMargin, id_).TechnicalMargin)

        return sum(result)
                                                                      
    @property
    def weightedUnderlyingAggregatedTM(self) -> float:
        result = []
        for gic in self.underlyingGic:
            id_ = dataclasses.replace(self.Identity)
            id_.DataNode = gic
            result.append(self.GetStorage().GetReinsuranceCoverage(self.Identity, gic) * self.GetScope(AllocateTechnicalMargin, id_).AggregatedTechnicalMargin)

        return sum(result)

    def ComputeReinsuranceEstimateType(self, aggregatedFcf: float) -> str:
        return  EstimateTypes.LR if aggregatedFcf > Precision else EstimateTypes.C
    
    @property
    def ComputedEstimateType(self) -> str:
        return self.ComputeReinsuranceEstimateType(self.weightedUnderlyingAggregatedTM + self.weightedUnderlyingTM)

     # In common2

    @property
    def balancingValue(self) -> float:

        result = {}
        for x in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Cashflow)).Values:
            result.setdefault(x.Novelty, []).append(x)

        temp = [g[-1] for g in result.values()]
        result = []
        for aoc in temp:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            result.append(
                self.GetScope(AllocateTechnicalMargin, id_).TechnicalMargin + self.GetScope(AllocateTechnicalMargin, id_).AggregatedTechnicalMargin if (
                        self.GetScope(AllocateTechnicalMargin, id_).ComputedEstimateType != self.ComputedEstimateType) else 0
            )

        return sum(result)
                                                   
    @property
    def HasSwitch(self) -> bool:
        return abs(self.balancingValue) > Precision

    @property
    def AggregatedTechnicalMargin(self) -> float:
        return self.balancingValue



class AllocateTechnicalMarginForCl(AllocateTechnicalMargin):

    @property
    def balancingValue(self) -> float:

        result = {}
        for x in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Cashflow)).Values:
            result.setdefault(x.Novelty, []).append(x)

        temp = [g[-1] for g in result.values()]
        result = []
        for aoc in temp:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            result.append(
                self.GetScope(AllocateTechnicalMargin, id_).TechnicalMargin + self.GetScope(AllocateTechnicalMargin, id_).AggregatedTechnicalMargin if (
                        self.GetScope(AllocateTechnicalMargin, id_).ComputedEstimateType != self.ComputedEstimateType) else 0
            )
        return sum(result)

    @property
    def HasSwitch(self) -> bool:
        return abs(self.balancingValue) > Precision
    
    @property
    def AggregatedTechnicalMargin(self) -> float:
        return self.balancingValue


class AllocateTechnicalMarginForBop(AllocateTechnicalMargin):

    @property
    def HasSwitch(self) -> bool:
        return False


class AllocateTechnicalMarginForEop(AllocateTechnicalMargin):

    @property
    def Value(self) -> float:
        result = []
        for aoc in self.GetScope(PreviousAocSteps, IdentityTuple3(self.Identity, InputSource.Cashflow)).Values:
            id_ = dataclasses.replace(self.Identity)
            id_.AocType = aoc.AocType
            id_.Novelty = aoc.Novelty
            result.append(self.GetScope(AllocateTechnicalMargin, id_).Value)

        return sum(result)

    @property
    def ComputedEstimateType(self) -> str:
        return self.ComputeEstimateType(self.AggregatedTechnicalMargin)


AllocateTechnicalMargin.Applicability = {
    AllocateTechnicalMarginForReinsuranceCL: lambda x: x.Identity.IsReinsurance and x.Identity.AocType == AocTypes.CL,
    AllocateTechnicalMarginForReinsurance: lambda x: x.Identity.IsReinsurance,   #  p => p.ForMember(s => s.ComputedEstimateType).ForMember(s => s.HasSwitch))
    ComputeAllocateTechnicalMarginWithIfrsVariable: lambda x: x.GetStorage().IsSecondaryScope(x.Identity.DataNode),
    AllocateTechnicalMarginForBop: lambda x: x.Identity.AocType == AocTypes.BOP,
    AllocateTechnicalMarginForCl: lambda x: x.Identity.AocType == AocTypes.CL,
    AllocateTechnicalMarginForEop: lambda x: x.Identity.AocType == AocTypes.EOP
}


class ContractualServiceMargin(IScope):      #<ImportIdentity, ImportStorage>

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.C
    
    @property
    def Value(self):
        return -1 * self.GetScope(AllocateTechnicalMargin, self.Identity, self.EstimateType).Value


class LossComponent(IScope):     #<ImportIdentity, ImportStorage>

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.L
    
    @property
    def Value(self) -> float:
        return self.GetScope(AllocateTechnicalMargin, self.Identity, self.EstimateType).Value


class LossRecoveryComponent(IScope):     #<ImportIdentity, ImportStorage>

    @property
    def EstimateType(self) -> str:
        return EstimateTypes.LR

    @property
    def Value(self) -> float:
        return self.GetScope(AllocateTechnicalMargin, self.Identity, self.EstimateType).Value


class DeferrableToIfrsVariable(IScope):   #<ImportIdentity, ImportStorage>

    @property
    def DeferrableActual(self) -> [IfrsVariable]:

        x = self.GetScope(DeferrableActual, self.Identity)

        if abs(x.Value) >= Precision:
            return [IfrsVariable(
                Id=uuid.uuid4(),
                EstimateType=x.EstimateType,
                 DataNode=x.Identity.DataNode,
                 AocType=x.Identity.AocType,
                 Novelty=x.Identity.Novelty,
                 AccidentYear=None,
                 Value=x.Value,
                 Partition=self.GetStorage().TargetPartition
                 )]
        else:
            return []


class EaForPremiumToIfrsVariable(IScope):  #<ImportIdentity, ImportStorage>

    @property
    def BeEAForPremium(self) -> [IfrsVariable]:

        if self.GetStorage().DataNodeDataBySystemName[self.Identity.DataNode].LiabilityType == LiabilityTypes.LIC or self.Identity.IsReinsurance:
            return []
        else:
            result = []
            for sc in self.GetScope(BeExperienceAdjustmentForPremium, self.Identity).ByAmountType:
                if abs(sc.Value) >= Precision:
                     result.append( IfrsVariable(
                         Id=uuid.uuid4(),
                        EstimateType = self.GetScope(BeExperienceAdjustmentForPremium, self.Identity).EstimateType,
                        DataNode = sc.Identity.Id.DataNode,
                        AocType = sc.Identity.Id.AocType,
                        Novelty = sc.Identity.Id.Novelty,
                        AccidentYear = sc.Identity.AccidentYear,
                        EconomicBasis = sc.EconomicBasis,
                        AmountType = sc.Identity.AmountType,
                        Value = sc.Value,
                        Partition = sc.GetStorage().TargetPartition ))

        return result

    @property
    def ActEAForPremium(self) -> [IfrsVariable]:

        if self.GetStorage().DataNodeDataBySystemName[self.Identity.DataNode].LiabilityType == LiabilityTypes.LIC or self.Identity.IsReinsurance:
            return []
        else:
            result = []
            for sc in self.GetScope(ActualExperienceAdjustmentOnPremium, self.Identity).ByAmountTypeAndEstimateType:
                if abs(sc.Value) >= Precision:
                                result.append(IfrsVariable(
                                    EstimateType = self.GetStorage().ExperienceAdjustEstimateTypeMapping[sc.Identity.EstimateType],
                                                 DataNode = sc.Identity.Id.DataNode,
                                                 AocType = sc.Identity.Id.AocType,
                                                 Novelty = sc.Identity.Id.Novelty,
                                                 AccidentYear = sc.Identity.AccidentYear,
                                                 #EconomicBasis = scope.EconomicBasis,
                                                 AmountType = sc.Identity.AmountType,
                                                 Value = sc.Value,
                                                 Partition = self.GetStorage().TargetPartition))

        return result


class TmToIfrsVariable(IScope):    #<ImportIdentity, ImportStorage>

    @property
    def EconomicBasis(self) -> str:
        return EconomicBases.C if self.Identity.ValuationApproach == ValuationApproaches.VFA else EconomicBases.L

    @property
    def AmortizationFactor(self) -> list[IfrsVariable]:

        if self.Identity.AocType == AocTypes.AM:

            result = []
            x = self.GetScope(CurrentPeriodAmortizationFactor, self.Identity, self.EconomicBasis)
            if abs(x.Value) >= Precision:
                result.append(IfrsVariable(
                    Id=uuid.uuid4(),
                    AmountType=None,
                    AccidentYear=None,
                    EstimateType = x.EstimateType,
                    DataNode = x.Identity.DataNode,
                    AocType = x.Identity.AocType,
                    Novelty = x.Identity.Novelty,
                    EconomicBasis = x.EconomicBasis,
                    Value = x.Value,
                    Partition = self.GetStorage().TargetPartition
                    ))
            return result
        else:
            return []


    @property
    def Csms(self) -> list[IfrsVariable]:

        if self.GetStorage().DataNodeDataBySystemName[self.Identity.DataNode].LiabilityType == LiabilityTypes.LIC:
            return []
        else:
            result = []
            x = self.GetScope(ContractualServiceMargin, self.Identity)
            if abs(x.Value) >= Precision:
                result.append(IfrsVariable(
                    Id=uuid.uuid4(),
                    AmountType=None,
                    AccidentYear=None,
                    EconomicBasis=None,
                    EstimateType = x.EstimateType,
                                   DataNode = x.Identity.DataNode,
                                   AocType = x.Identity.AocType,
                                   Novelty = x.Identity.Novelty,
                                   Value = x.Value,
                                   Partition = self.GetStorage().TargetPartition)
                                )
            return result


    @property
    def Loss(self) -> list[IfrsVariable]:
        if self.GetStorage().DataNodeDataBySystemName[self.Identity.DataNode].LiabilityType == LiabilityTypes.LIC:
            return []

        else:
            if self.Identity.IsReinsurance:
                result = []
                x = self.GetScope(LossRecoveryComponent, self.Identity)
                if abs(x.Value)>= Precision:
                    result.append(IfrsVariable(
                        Id=uuid.uuid4(),
                        AmountType=None,
                        AccidentYear=None,
                        EconomicBasis=None,
                        EstimateType = x.EstimateType,
                       DataNode = x.Identity.DataNode,
                       AocType = x.Identity.AocType,
                       Novelty = x.Identity.Novelty,
                       Value = x.Value,
                       Partition = self.GetStorage().TargetPartition

                    ))
            else:
                result = []
                x = self.GetScope(LossComponent, self.Identity)
                if abs(x.Value) >= Precision:
                   result.append(
                       IfrsVariable(
                           Id=uuid.uuid4(),
                           AmountType=None,
                           AccidentYear=None,
                           EconomicBasis=None,
                           EstimateType = x.EstimateType,
                           DataNode = x.Identity.DataNode,
                           AocType = x.Identity.AocType,
                           Novelty = x.Identity.Novelty,
                           Value = x.Value,
                           Partition = self.GetStorage().TargetPartition
                                                            ))

            return result



class ActualToIfrsVariable(IScope):    #<ImportIdentity, ImportStorage>

    @property
    def Actual(self) -> list[IfrsVariable]:

        result = []
        for x in self.GetScope(Actual, self.Identity).Actuals:
            if abs(x.Value) >= Precision:
                result.append(IfrsVariable(
                    EstimateType = x.Identity.EstimateType,
                                   DataNode = x.Identity.Id.DataNode,
                                   AocType = x.Identity.Id.AocType,
                                   Novelty = x.Identity.Id.Novelty,
                                   AccidentYear = x.Identity.AccidentYear,
                                   AmountType = x.Identity.AmountType,
                                   Value = x.Value,
                                   Partition = self.GetStorage().TargetPartition
                                            ))

        return result

    @property
    def AdvanceActual(self) -> list[IfrsVariable]:
        result = []
        for x in self.GetScope(AdvanceActual, self.Identity).Actuals:
            if abs(x.Value) >= Precision:
                result.append(IfrsVariable( EstimateType = x.Identity.EstimateType,
                                   DataNode = x.Identity.Id.DataNode,
                                   AocType = x.Identity.Id.AocType,
                                   Novelty = x.Identity.Id.Novelty,
                                   AccidentYear = x.Identity.AccidentYear,
                                   AmountType = x.Identity.AmountType,
                                   Value = x.Value,
                                   Partition = self.GetStorage().TargetPartition
                ))

        return result


    @property
    def OverdueActual(self) -> list[IfrsVariable]:
        result = []
        for x in self.GetScope(OverdueActual, self.Identity).Actuals:
            if abs(x.Value) >= Precision:
                result.append(IfrsVariable(
                    EstimateType = x.Identity.EstimateType,
                                   DataNode = x.Identity.Id.DataNode,
                                   AocType = x.Identity.Id.AocType,
                                   Novelty = x.Identity.Id.Novelty,
                                   AccidentYear = x.Identity.AccidentYear,
                                   AmountType = x.Identity.AmountType,
                                   Value = x.Value,
                                   Partition = self.GetStorage().TargetPartition

                ))
        return result


class ComputeIfrsVarsCashflows(
    PvToIfrsVariable, RaToIfrsVariable, DeferrableToIfrsVariable, EaForPremiumToIfrsVariable, TmToIfrsVariable):


    @property
    def CalculatedIfrsVariables(self) -> list[IfrsVariable]:
        return (self.PvLocked + self.PvCurrent + self.RaCurrent
                + self.RaLocked + self.AmortizationFactor
                + self.BeEAForPremium + self.DeferrableActual
                + self.Csms + self.Loss)


class ComputeIfrsVarsActuals(ActualToIfrsVariable, DeferrableToIfrsVariable, EaForPremiumToIfrsVariable, TmToIfrsVariable):

    @property
    def CalculatedIfrsVariables(self) -> list[IfrsVariable]:
        return self.Actual + self.AdvanceActual + self.OverdueActual + self.ActEAForPremium + self.DeferrableActual + self.Csms + self.Loss


class ComputeIfrsVarsOpenings(ActualToIfrsVariable, DeferrableToIfrsVariable, TmToIfrsVariable):


    @property
    def CalculatedIfrsVariables(self) -> list[IfrsVariable]:
        return self.AdvanceActual + self.OverdueActual + self.DeferrableActual + self.Csms + self.Loss



