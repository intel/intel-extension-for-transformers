import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

new_summary_file = sys.argv[1]
last_summary_file = sys.argv[2]
PCT_COL_NAME = r"(new/last)%"
THRESHOLD_REGRESSION = 80
RdYlGn = plt.cm.get_cmap('RdYlGn')
name_no_ext = new_summary_file.split('/')[-1][:-4]
xlsx_path = "/intel-extension-for-transformers/benchmark_log"


def to_df(fname):
    try:
        df = pd.read_csv(fname, sep=";", na_values='na')
        df = df.iloc[:, :-1]
        df.set_index([c for c in df.columns if not c.startswith(
            "Unnamed") and not c in ["acc", "perf"]], inplace=True)
        df[["time", "gflops"]] = df["perf"].str.split(",", expand=True)
        df["time"] = pd.to_numeric(df["time"])
        df["gflops"] = pd.to_numeric(df["gflops"])
        df.drop(["perf"], axis=1, inplace=True)
    except:
        df = pd.DataFrame()
    return df


def sync_idx_name(df_target: pd.DataFrame, df_source: pd.DataFrame):
    if (df_source.shape == (0, 0)):
        return
    for iname in df_source.index.names:
        if iname not in df_target.index.names:
            dtype = df_source.index.dtypes[iname]
            df_target.loc[:, iname] = np.array(
                0 if dtype in [np.int32, np.int64] else None, dtype=dtype)
            df_target.set_index(iname, append=True, inplace=True)


df_new = to_df(new_summary_file)
df_last_raw = to_df(last_summary_file)

# sync index names
if df_last_raw.shape == (0, 0):
    df_last_raw = pd.DataFrame(index=df_new.index, columns=df_new.columns)
sync_idx_name(df_new, df_last_raw)
sync_idx_name(df_last_raw, df_new)
df_last_raw = df_last_raw.reorder_levels(df_new.index.names)

# need to make sure their idx is identical
df_last = pd.DataFrame(index=df_new.index, columns=df_new.columns)
for idx, val in df_last_raw.iterrows():
    df_last.loc[idx] = val
    if idx not in df_new.index:
        df_new.loc[idx] = pd.Series(dtype=object)

df_comp = df_new.compare(df_last, keep_equal=True, keep_shape=True)
df_comp.rename(columns={
    "self": "new",
    "other": "last",
}, inplace=True)

df_comp.loc[:, ("time", PCT_COL_NAME)] = df_comp.loc[:,
                                                     ("time", "new")] / df_comp.loc[:, ("time", "last")] * 100
df_comp.loc[:, ("gflops", PCT_COL_NAME)] = df_comp.loc[:,
                                                       ("gflops", "new")] / df_comp.loc[:, ("gflops", "last")] * 100
df_comp = df_comp.loc[:, ["time", "gflops"]]

# set style
idx_pct = pd.IndexSlice[:, pd.IndexSlice[:, PCT_COL_NAME]]
idx_pct_time = pd.IndexSlice[:, pd.IndexSlice['time', PCT_COL_NAME]]
idx_pct_gflops = pd.IndexSlice[:, pd.IndexSlice['gflops', PCT_COL_NAME]]
idx_time = pd.IndexSlice[:, pd.IndexSlice['time', :]]
idx_gflops = pd.IndexSlice[:, pd.IndexSlice['gflops', :]]
df_comp_styler = df_comp.style \
    .format('{:0.2f}', na_rep='-')\
    .format('{:0.4f}', subset=idx_time, na_rep='-')\
    .format('{:0.2f}%', subset=idx_pct, na_rep='-')\
    .format_index({df_comp.index.get_level_values(i).name: '{:.2f}' for i in range(len(df_comp.index.levels))
                   if df_comp.index.get_level_values(i).dtype in [np.float32, np.float64]}, axis=0)\
    .background_gradient(RdYlGn.reversed(), subset=idx_pct_time, vmin=80, vmax=120)\
    .background_gradient(RdYlGn, subset=idx_pct_gflops, vmin=80, vmax=120)\
    .set_table_attributes('class="features-table"')
df_comp_styler.to_excel(f"{xlsx_path}/{name_no_ext}.xlsx")

# output report via stdout
table_repr = df_comp_styler.to_html()
geo_mean_repr = np.exp(df_comp.applymap(np.log).mean())
geo_mean_repr = '\n'.join(
    str(geo_mean_repr).split('\n')[:-1])  # remove the last line showing dtype
geo_mean_repr = f"<pre>GeoMean:\n{geo_mean_repr}</pre>"
print(
    f"<div class='summary-wrapper {name_no_ext}'>{table_repr}{geo_mean_repr}</div>")

# ouput regression message via stderr
min_gflops_pct = df_comp.loc[:, ("gflops", PCT_COL_NAME)].min(skipna=True)
if (min_gflops_pct < THRESHOLD_REGRESSION):
    print(
        f"Min GFLOPS ({min_gflops_pct:.2f}%) is less than {THRESHOLD_REGRESSION}% in {new_summary_file}", file=sys.stderr)
