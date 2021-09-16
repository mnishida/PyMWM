import os
import subprocess

from pymwm.waveguide import Database


def main():
    dirname = os.path.join(os.path.expanduser("~"), ".pymwm")
    rootname = os.path.join(dirname, "pymwm_data")

    subprocess.call(f"cp {rootname}.h5 {rootname}.h5.old", shell=True)

    db = Database()
    catalog = db.load_catalog()

    catalog = catalog.replace(
        {
            "gold_dl": "Au Stewart-DLF",
            "gold_rakic": "Au Rakic-DLF",
            "gold_d": "Au Vial-DF",
            "silver_dl": "Ag Vial-DLF",
            "aluminium_dl": "Al Rakic-DLF",
            "pec": "PEC",
        }
    )
    catalog["core"] = catalog["core"].str.replace("RI_", "RI: ")
    for idx in catalog.query("im_factor == 0.0").index:
        catalog.loc[idx, "clad"] = catalog.loc[idx, "clad"] + " im_factor: 0.0"
    catalog = catalog.drop("im_factor", axis=1)

    print(catalog)
    catalog.to_hdf(f"{rootname}.h5", "catalog")


if __name__ == "__main__":
    main()
