import numpy as np
import pandas as pd


def test_present_values():

    from ifrs17_template.template import df as actual

    actual.replace({float('nan'): None}, inplace=True)
    expected = pd.read_excel(
        "results/PresentValues.xlsx", index_col=0, dtype={'Value': float}
    )
    expected.replace({float('nan'): None}, inplace=True)
    cols = [c for c in expected.columns if c not in ('Value', 'AccidentYear', 'OciType', 'YieldCurveName', 'Partner')]

    assert (actual.loc[:, cols]).equals(expected.loc[:, cols])
    assert np.all(np.isclose(pd.to_numeric(actual.Value.values), pd.to_numeric(expected.Value.values)))