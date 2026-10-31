from dagster import Definitions, load_assets_from_modules
from lease_intel.assets import bronze, silver, gold

defs = Definitions(
    assets=[
        *load_assets_from_modules([bronze]),
        *load_assets_from_modules([silver]),
        *load_assets_from_modules([gold]),
    ]
)
