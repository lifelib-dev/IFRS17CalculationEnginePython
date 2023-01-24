import numpy as np
import pandas as pd


def test_present_values():

    from PresentValueSeries.PresentValueEps2 import get_ifrsvars, DataSource

    actual = get_ifrsvars(DataSource)
    expected = pd.read_excel(
        "results/PresentValueEp2.xlsx", index_col=0, dtype={'Value': float}
    )
    expected['AccidentYear'].replace({float('nan'): None}, inplace=True)
    cols = [c for c in expected.columns if c != 'Value']
    assert (actual.loc[:, cols]).equals(expected.loc[:, cols])
    assert np.all(np.isclose(actual.Value.values, expected.Value.values))


