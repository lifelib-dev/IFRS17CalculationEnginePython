from functools import total_ordering
import dataclasses
from dataclasses import dataclass
from .Systemorph import *
from .Consts import *
from .Enums import *
from .Validations import *


# Data Infrastructure

## Base Interfaces




@dataclass
class IKeyed(metaclass=IKeyedType):
    Id: Guid


class IPartition(IKeyed):
    pass


@dataclass
class IPartitioned:
    Partition: Guid


@dataclass
class IHierarchy:
    Name: str
    Parent: str
    Child: str


@dataclass
class IWithYearAndMonth:
    Year: int
    Month: int


@dataclass
class IWithYearMonthAndScenario(IWithYearAndMonth):
    Scenario: str # = dataclasses.field(default='', init=False)

    # def __post_init__(self, Scenario):
    #     self.Scenario = Scenario


## Abstract Classes

class KeyedRecord(IKeyed):
    pass



@dataclass
class KeyedDimension(INamed):
    SystemName: str
    DisplayName: str

    # def __hash__(self):
    #     return hash(self.SystemName)
    #
    # def __eq__(self, other):
    #     return self.SystemName == other.SystemName

@dataclass
class KeyedOrderedDimension(KeyedDimension, IOrdered):
    Order: int


@dataclass
class KeyedOrderedDimensionWithExternalId(KeyedOrderedDimension):
    ExternalId: list[str]


# Dimensions


@dataclass
class HierarchicalDimensionWithLevel(IHierarchicalDimension):
    SystemName: str
    DisplayName: str
    Parent: str
    Level: int


## Amount Type

@dataclass
class AmountType(KeyedOrderedDimensionWithExternalId, IHierarchicalDimension):
    Parent: str
    PeriodType: PeriodType


class DeferrableAmountType(AmountType):
    pass


## Risk Driver
@dataclass
class RiskDriver(KeyedOrderedDimension, IHierarchicalDimension):
    Parent: str


## Estimate Type
@dataclass
class EstimateType(KeyedOrderedDimensionWithExternalId): 
    InputSource: InputSource
    StructureType: StructureType
    PeriodType: PeriodType


## Novelty
class Novelty(KeyedOrderedDimension):
    pass


## Variable Type
@dataclass
class VariableType(KeyedOrderedDimension, IHierarchicalDimension): 
    Parent: str


### AoC Variable Type
class AocType(VariableType): 
    pass


@dataclass
class AocStep:
    AocType: str
    Novelty: str

    def __hash__(self):
        return hash((self.AocType, self.Novelty))

    def __eq__(self, other):
        return self.AocType == other.AocType and self.Novelty == other.Novelty


class PnlVariableType(VariableType):
    pass


class BsVariableType(VariableType):
    pass


class AccountingVariableType(VariableType):
    pass


## Scenario
class Scenario(KeyedDimension):
    pass


## Line Of Business
@dataclass
class LineOfBusiness(KeyedOrderedDimension, IHierarchicalDimension): 
    Parent: str


## Currency
class Currency(KeyedDimension):
    pass


## Economic Basis
class EconomicBasis(KeyedDimension):
    pass


## Valuation Approach
class ValuationApproach(KeyedDimension):
    pass


## Liability Type
@dataclass
class LiabilityType(KeyedDimension, IHierarchicalDimension): 
    Parent: str


## OCI Type

class OciType(KeyedDimension):
    pass


## Profitability

class Profitability(KeyedDimension):
    pass

## Partner


class Partner(KeyedDimension):
    pass

## Credit Risk Rating


class CreditRiskRating(KeyedDimension):
    pass

## Reporting Node

@dataclass
class ReportingNode(KeyedDimension, IHierarchicalDimension):
    Parent: str
    Currency: str


@dataclass
class ProjectionConfiguration(KeyedDimension):
    Shift: int
    TimeStep: int


@dataclass
class AocConfiguration(KeyedRecord, IWithYearAndMonth, IOrdered):

    Year: int
    Month: int
    AocType: str
    Novelty: str
    DataType: DataType
    InputSource: InputSource
    FxPeriod: FxPeriod
    YcPeriod: PeriodType
    CdrPeriod: PeriodType
    ValuationPeriod: ValuationPeriod
    RcPeriod: PeriodType
    Order: int


@dataclass
class ExchangeRate(KeyedRecord, IWithYearMonthAndScenario):

    Currency: str
    Year: int
    Month: int
    FxType: FxType
    FxToGroupCurrency: float
    # Scenario: str = ''


@dataclass
class CreditDefaultRate(KeyedRecord, IWithYearMonthAndScenario): 

    CreditRiskRating: str
    Year: int
    Month: int
    Values: list[float]
    # Scenario: str = ''


@dataclass
class YieldCurve(KeyedRecord, IWithYearMonthAndScenario):

    Currency: str
    Year: int
    Month: int
    Name: str
    Values: list[float]
    # Scenario: str = ''

@dataclass
class PartnerRating(KeyedRecord, IWithYearMonthAndScenario): 

    Partner: str
    CreditRiskRating: str
    Year: int
    Month: int
    # Scenario: str = ''


# Partitions
@dataclass
class IfrsPartition(IPartition):
    ReportingNode: str
    Scenario: str = ''


class PartitionByReportingNode(IfrsPartition):
    pass


@total_ordering
@dataclass
class PartitionByReportingNodeAndPeriod(IfrsPartition):
    Year: int = 0
    Month: int = 0

    def __hash__(self):
        return hash((self.Id, self.ReportingNode, self.Scenario, self.Year, self.Month))

    def __eq__(self, other):
        return (self.Id, self.ReportingNode, self.Scenario, self.Year, self.Month) == (
            other.Id, other.ReportingNode, other.Scenario, other.Year, other.Month)

    def __lt__(self, other):
        return ((self.Id, self.ReportingNode, self.Scenario, self.Year, self.Month) < (
            other.Id, other.ReportingNode, other.Scenario, other.Year, other.Month))


# Policy-related Data Structures


@dataclass
class DataNode(KeyedDimension, IPartitioned): 

    Partition: Guid
    ContractualCurrency: str
    FunctionalCurrency: str
    LineOfBusiness: str
    ValuationApproach: str
    OciType: str


@dataclass
class Portfolio(DataNode): 
    pass


@dataclass
class InsurancePortfolio(Portfolio):
    pass 


@dataclass
class ReinsurancePortfolio(Portfolio):
    pass


@dataclass(eq=False)
class GroupOfContract(DataNode):

    AnnualCohort: int
    LiabilityType: str
    Profitability: str
    Portfolio: str
    YieldCurveName: str
    Partner: str


@dataclass(eq=False)
class GroupOfInsuranceContract(GroupOfContract): 
    pass
    # [Immutable]

    #  TODO: for the case of internal reinsurance the Partner would be the reporting node, hence not null.
    #  If this is true we need the [Required] attribute here, add some validation at dataNode import 
    #  and to add logic in the GetNonPerformanceRiskRate method in ImportStorage.


@dataclass(eq=False)
class GroupOfReinsuranceContract(GroupOfContract):
    pass


@dataclass
class DataNodeState(KeyedRecord, IPartitioned, IWithYearMonthAndScenario): 

    Partition: Guid
    DataNode: str
    Year: int
    Month: int      # = DefaultDataNodeActivationMonth
    State: State    # = State.Active
    # Scenario: str = ''


@dataclass
class DataNodeParameter(KeyedRecord, IPartitioned, IWithYearMonthAndScenario):

    Partition: Guid
    Year: int
    Month: int  # = DefaultDataNodeActivationMonth
    DataNode: str
    # Scenario: str


@dataclass
class SingleDataNodeParameter(DataNodeParameter):
    PremiumAllocation: float = DefaultPremiumExperienceAdjustmentFactor


@dataclass
class InterDataNodeParameter(DataNodeParameter):
    LinkedDataNode: str
    ReinsuranceCoverage: float
    Scenario: str

    def __hash__(self):
        return hash((self.LinkedDataNode, self.ReinsuranceCoverage, self.Scenario))

    def __eq__(self, other):
        return (self.LinkedDataNode, self.ReinsuranceCoverage, self.Scenario) == (
            other.LinkedDataNode, other.ReinsuranceCoverage, other.Scenario)


@dataclass
class DataNodeData:

    DataNode: str

    # Portfolio

    ContractualCurrency: str
    FunctionalCurrency: str
    LineOfBusiness: str
    ValuationApproach: str
    OciType: str

    # GroupOfContract

    Portfolio: str
    AnnualCohort: int
    LiabilityType: str
    Profitability: str
    Partner: str

    # DataNodeState

    Year: int
    Month: int
    State: State
    PreviousState: State
    IsReinsurance: bool
    Scenario: str


## Variables

@dataclass
class BaseVariableIdentity:
    DataNode: str
    AocType: str
    Novelty: str


@dataclass
class BaseDataRecord(BaseVariableIdentity, IKeyed, IPartitioned):
    AmountType: str
    AccidentYear: int


@dataclass
class RawVariable(BaseDataRecord):
    Values: list[float]
    EstimateType: str


@total_ordering
@dataclass
class IfrsVariable(BaseDataRecord):
    Value: float
    EstimateType: str
    EconomicBasis: str

    def __hash__(self):
        return hash((self.DataNode, self.AocType, self.Novelty, self.AmountType, self.AccidentYear, self.EstimateType, self.EconomicBasis))

    def __eq__(self, other):
        if eq := self.Id == other.Id:
            assert hash(self) == hash(other)
        return eq

    def __lt__(self, other):
        return ((self.DataNode, self.AocType, self.Novelty, self.AmountType, self.AccidentYear, self.EstimateType, self.EconomicBasis, self.Id) < (
            other.DataNode, other.AocType, other.Novelty, other.AmountType, other.AccidentYear, other.EstimateType, other.EconomicBasis, self.Id))

# Import Identity

@total_ordering
@dataclass
class ImportIdentity(BaseVariableIdentity):

    IsReinsurance: bool = False
    ValuationApproach: str = ''
    ProjectionPeriod: int = 0
    ImportScope: ImportScope = None

    def __hash__(self):
        return hash((self.DataNode, self.AocType, self.Novelty))

    def __eq__(self, other):
        return (self.DataNode, self.AocType, self.Novelty) == (
            other.DataNode, other.AocType, other.Novelty)

    def __lt__(self, other):
        return ((self.DataNode, self.AocType, self.Novelty) < (
            other.DataNode, other.AocType, other.Novelty))

    @property
    def AocStep(self) -> (str, str):
        return self.AocType, self.Novelty

    @classmethod
    def from_iv(cls, iv) -> 'ImportIdentity':
        return cls(
            DataNode=iv.DataNode,
            AocType=iv.AocType,
            Novelty=iv.Novelty)

    @classmethod
    def from_rv(cls, rv) -> 'ImportIdentity':
        return cls(
            DataNode=rv.DataNode,
            AocType=rv.AocType,
            Novelty=rv.Novelty)



# Args

@dataclass
class Args:
    ReportingNode: str
    Year: int
    Month: int
    Periodicity: Periodicity
    Scenario: str


@dataclass
class ImportArgs(Args):
    ImportFormat: str
