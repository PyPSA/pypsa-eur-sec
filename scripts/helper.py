import os
import pandas as pd
from pypsa.descriptors import Dict
from pypsa.components import components, component_attrs

import logging
logger = logging.getLogger(__name__)


def override_component_attrs(directory):
    """Tell PyPSA that links can have multiple outputs by
    overriding the component_attrs. This can be done for
    as many buses as you need with format busi for i = 2,3,4,5,....
    See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs

    Parameters
    ----------
    directory : string
        Folder where component attributes to override are stored 
        analogous to ``pypsa/component_attrs``, e.g. `links.csv`.

    Returns
    -------
    Dictionary of overriden component attributes.
    """

    attrs = Dict({k : v.copy() for k,v in component_attrs.items()})

    for component, list_name in components.list_name.items():
        fn = f"{directory}/{list_name}.csv"
        if os.path.isfile(fn):
            overrides = pd.read_csv(fn, index_col=0, na_values="n/a")
            attrs[component] = overrides.combine_first(attrs[component])

    return attrs