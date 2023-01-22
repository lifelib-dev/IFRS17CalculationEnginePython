
def GetElementOrDefault(array: list, index: int):

    count = len(array)
    if not array:
        return 0

    return array[index] if index < count else array[count-1]


def AggregateDoubleArray(source):

    if not (size := max((len(nested) for nested in source), default=0)):
        return []

    result = [0] * size

    for nested in source:
        for i in range(len(nested)):
            result[i] += nested[i]

    return result


# public static double[] AggregateDoubleArray(this IEnumerable<IEnumerable<double>> source) => source.Where(x => x is not null).DefaultIfEmpty(Enumerable.Empty<double>()).Aggregate((x, y) => x.ZipLongest(y, (a, b) => a + b)).ToArray();