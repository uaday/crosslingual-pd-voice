"""Dataset loaders — each returns the standard manifest DataFrame (see base.py).

Use the LOADERS registry to stay dataset-agnostic:

    from pdvoice.data import LOADERS
    df = LOADERS["ipvs"]()
"""

from pdvoice.data.bridge2ai import load_bridge2ai
from pdvoice.data.ewadb import load_ewadb
from pdvoice.data.ipvs import load_ipvs
from pdvoice.data.mdvr_kcl import load_mdvr_kcl
from pdvoice.data.mpower import load_mpower
from pdvoice.data.neurovoz import load_neurovoz

LOADERS = {
    "mpower": load_mpower,
    "bridge2ai": load_bridge2ai,
    "mdvr_kcl": load_mdvr_kcl,
    "ipvs": load_ipvs,
    "neurovoz": load_neurovoz,
    "ewadb": load_ewadb,
}

__all__ = [
    "LOADERS",
    "load_mpower",
    "load_bridge2ai",
    "load_mdvr_kcl",
    "load_ipvs",
    "load_neurovoz",
    "load_ewadb",
]
