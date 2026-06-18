import csv
import numpy as np

def read_radionuclides_file(file_name):
    radionuclides = []
    half_life_values = []
    units = []
    types = []

    with open(file_name, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            line = ",".join(row).strip()
            if not line or line.startswith("#"):
                continue

            if row[0].strip() == "Radionuclido":
                continue

            if len(row) >= 4:
                radionuclides.append(row[0].strip())
                half_life_values.append(row[1].strip())
                units.append(row[2].strip())
                types.append(row[3].strip())

    return radionuclides, half_life_values, units, types

from importlib.resources import files

def lambda_for_species(species_name):
    path = files("radioprotection.data").joinpath("radionuclidos.csv")
    radionuclides, half_life_values, units, types = read_radionuclides_file(path)

    target_position = radionuclides.index(species_name)
    half_life = float(half_life_values[target_position])
    unit = units[target_position]

    if unit == "years":
        return np.log(2) / (half_life * 365.25 * 24 * 3600)
    elif unit == "days":
        return np.log(2) / (half_life * 24 * 3600)
    elif unit == "hours":
        return np.log(2) / (half_life * 3600)
    elif unit == "minutes":
        return np.log(2) / (half_life * 60)
    elif unit == "seconds":
        return np.log(2) / half_life