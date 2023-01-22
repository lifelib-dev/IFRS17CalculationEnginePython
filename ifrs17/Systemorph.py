import dataclasses
import uuid
from dataclasses import dataclass
from typing import (
    TypeVar, Generic, Iterable, Optional, Union, Callable,
    Dict, Collection)


IdentityType = TypeVar('IdentityType')
StorageType = TypeVar('StorageType')

IEnumerable = Iterable

Guid = uuid.UUID


class INamed:
    pass


class IOrdered:
    pass

@dataclass
class IScope(Generic[IdentityType, StorageType]):
    Identiry: IdentityType


@dataclass
class IDataSet:
    Tables: dict[str, Generic]


class IDataRow:
    pass


class IKeyedType(type):
    pass


class IHierarchicalDimension:
    pass


class IHierarchicalDimensionCache:

    def InitializeAsync(self, type_: type):
        pass


class IQuerySource:

    def __init__(self):

        self._data = {}
        self._current_partition = {}

    def Query(self, type_: type):
        result = []
        for k in self._data:
            if issubclass(k, type_):
                result.extend(self._data[k])

        return result

    def DeleteAsync(self, type_: type, data: Collection):
        records = self._data.get(type_, [])
        for r in records.copy():
            if r in data:
                records.remove(r)

    def UpdateAsync(self, type_: type, data: Collection):
        self._data.setdefault(type_, []).extend(data)

    def UpdateAsync2(self, data):
        self._data.setdefault(type(data), []).append(data)

    def UpdateAsync3(self, datalist):
        assert len(set(type(x) for x in datalist)) <= 1
        for x in datalist:
            self.UpdateAsync2(x)

    def CommitToTargetAsync(self, other: "IQuerySource"):
        other._data = self._data

    def InitializeFrom(self, other: "IQuerySource"):
        for k, v in other._data.items():
            self._data[k] = v

    def Initialize(self, source: "IQuerySource", DisableInitialization: list[type]):
        for k, v in source._data.items():
            if k in DisableInitialization:
                pass
            else:
                self._data[k] = v

    def ToHierarchicalDimensionCache(self) -> IHierarchicalDimensionCache:
        return IHierarchicalDimensionCache()

    class _Partition:

        def __init__(self, data: "IQuerySource"):
            self._querysource = data

        def GetKeyForInstanceAsync(self, PartitionType: IKeyedType, args: Generic):
            try:
                partitions = self._querysource.Query(PartitionType)
            except KeyError:
                return uuid.uuid4()

            fields = set(f.name for f in dataclasses.fields(PartitionType))
            fields.remove('Id')

            found = False
            for x in partitions:
                if all(getattr(args, f) == getattr(x, f) for f in fields):
                    found = True
                    break

            if found:
                return x.Id
            else:
                return uuid.uuid4()

        def SetAsync(self, PartitionType: IKeyedType, Id: Guid):
            self._querysource._current_partition[PartitionType.__name__] = Id

        def GetCurrent(self, TypeName: str):
            return self._querysource._current_partition[TypeName]


    @property
    def Partition(self):
        return self._Partition(self)




class IDataSource(IQuerySource):
    pass


DataSource = IDataSource()


class IWorkspace(IQuerySource):

    DataSource = DataSource


Workspace = IWorkspace()


class ActivityLog:
    pass



class Task:
    pass



