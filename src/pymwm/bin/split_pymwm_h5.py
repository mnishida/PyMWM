import os

import pandas as pd

from pymwm.waveguide import Database


def main():
    dirname = os.path.join(os.path.expanduser("~"), ".pymwm")
    dir_data = os.path.join(dirname, "data")
    filename = os.path.join(dirname, "pymwm_data.h5")

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    if not os.path.exists(dir_data):
        os.mkdir(dir_data)

    db = Database()
    catalog = pd.read_hdf(filename, "catalog")
    catalog_new = pd.read_hdf(filename, "catalog")

    sn_new = 0
    while len(catalog):
        cond = ""
        for col, t in list(db.catalog_columns.items())[1:-3]:
            val = catalog.iloc[0][f"{col}"]
            if t == str:
                cond += f"{col} == '{val}' & "
            else:
                cond += f"{col} == {val} & "
        cond = cond.rstrip("& ")
        for i, sn, EM, n, m in catalog.query(cond)[["sn", "EM", "n", "m"]].itertuples():
            df = pd.read_hdf(filename, f"sn_{sn}")
            df.to_hdf(
                os.path.join(dir_data, f"{sn_new:06}.h5"),
                f"{EM}_{n}_{m}",
                complevel=9,
                complib="blosc",
            )
            catalog.drop(i, inplace=True)
            catalog_new.loc[i, "sn"] = sn_new
        sn_new += 1
    db.save_catalog(catalog_new)


if __name__ == "__main__":
    main()
