import json
import os
from importlib.resources import files

def read_hrtm_data():
    data = []
    json_paths = ["hrtm_subjects_breathing.json", "dcf_public.json", "dcf_workers.json"]
    for path in json_paths:
        json_path = files("radioprotection.data").joinpath(path)

        with json_path.open("r", encoding="utf-8") as f:
            data.append(json.load(f))
    return data

def assign_gender_children(gender, age_group):
    if age_group in ["age_10y", "age_5y", "age_1y", "infant"]:
        return "default"
    return gender

def determine_breathing_rate(data, age_group, gender, physical_activity):
    return data[0]["subjects"][age_group][gender]["breathing"][physical_activity]["B_m3_h"]

def determine_inhalation_dose_coefficients(data, population_type, species_name, absorption, age_group):
    if population_type == "public":
        inhalation_dose_coefficients = float(data[1][species_name]["inhalation"][absorption][age_group]["e"])
    elif population_type == "worker":
        inhalation_dose_coefficients = float(data[2][species_name]["inhalation"][absorption][age_group]["e_1um"])
    else:
        raise ValueError(f"Tipo de población no válido: {population_type}")

    return inhalation_dose_coefficients

def determine_dose(concentration, breathing_rate, exposition_time, inhalation_dose_coefficients):
    dose = {}
    for key in concentration:
        dose[key] = concentration[key] * breathing_rate * exposition_time * inhalation_dose_coefficients
    return dose