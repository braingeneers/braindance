import pandas as pd
from IPython import embed
from braindance.config import get_catalog_path

cat = pd.read_csv(get_catalog_path())
embed()

# filter to see all the rows that have nothing in the num_units column
# make sure it shows the full output not truncated
pd.set_option("display.max_rows", None)
cat[cat["num_units"].isnull()]
