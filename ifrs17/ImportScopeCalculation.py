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
        temp = {rv.AmountType for rv in self.GetStorage().GetRawVariables(self.Identity) if rv.AmountType != ''}
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


class ReferenceAocStep(IScope) :  #IScope<ImportIdentity, ImportStorage>

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
        return -1 * ComputeDiscountAndCumulateWithCreditDefaultRisk(self.NominalValues, self.MonthlyDiscounting, self.nonPerformanceRiskRate)     # we need to flip the sign to create a reserve view


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
        identity = IdentityTuple(*self.Identity)
        importid = dataclasses.replace(self.Identity.Id)
        importid.AocType = AocTypes.CF
        identity.Id = importid

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

            interestOnClaimsCashflow[i] = 1 / GetElementOrDefault(self.monthlyInterestFactor, int(i/12)) * (interestOnClaimsCashflow[i + 1] + self.nominalClaimsCashflow[i] - (self.nominalClaimsCashflow[i + 1] if i+1 < len(self.nominalClaimsCashflow) else 0))
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
        return self.GetValueFromValues(self.Values)


class ComputePresentValueWithIfrsVariable(PresentValue):


    @property
    def Value(self) -> list[float]:
        return self.GetStorage().GetValue(
            self.Identity.Id, self.Identity.AmountType, self.Identity.EstimateType, EconomicBasis, self.Identity.AccidentYear)

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

            # if self.Identity.AocType == 'BOP' and self.Identity.Novelty == 'I':
            #     print('stop')

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


class ComputeIfrsVarsCashflows(
    PvToIfrsVariable):  #, RaToIfrsVariable, DeferrableToIfrsVariable, EaForPremiumToIfrsVariable, TmToIfrsVariable):


    @property
    def CalculatedIfrsVariables(self) -> list[IfrsVariable]:
        return self.PvLocked + self.PvCurrent

                             # .Concat(RaCurrent)
                             # .Concat(RaLocked)
                             # .Concat(AmortizationFactor)
                             # .Concat(BeEAForPremium)
                             # .Concat(DeferrableActual)
                             # .Concat(Csms)
                             # .Concat(Loss);


# public interface IModel : IMutableScopeWithStorage<ImportStorage>{}


@dataclass
class DeferrableActual(IScope[ImportIdentity, ImportStorage]):

    EstimateType: str = EstimateTypes.DA
    EconomicBasis: str = EconomicBases.L

    # @property
    # def Value(self):
    #     GetStorage().GetValue(self.Identity, (string), null, EstimateType, (int?) null)

"""    public
    double
    Value = > GetStorage().GetValue(Identity, (string)
    null, EstimateType, (int?)
    null);
"""

# class DeferrableToIfrsVariable(IScope[ImportIdentity, ImportStorage]):
#
#     @property
#     def DeferrableActual(self) -> list[IfrsVariable]:
#         x = DeferrableActual(self.Identiry)
#         if abs(x.Value) >= Precision:
#             return IfrsVariable(
#                 EstimateType=x.EstimateType
#             )
#         else:
#             return []



"""
    IEnumerable<IfrsVariable> DeferrableActual => GetScope<DeferrableActual>(Identity).RepeatOnce()
    .Where(x => Math.Abs(x.Value) >= Precision).Select(
    
    x => new IfrsVariable{ EstimateType = x.EstimateType,
                                    DataNode = x.Identity.DataNode,
                                    AocType = x.Identity.AocType,
                                    Novelty = x.Identity.Novelty,
                                    AccidentYear = null,
                                    Value = x.Value,
                                    Partition = GetStorage().TargetPartition
                                    });

"""