from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CEMS_CARB_INTENSITY",
    "CEMS_COL_MAP",
    "COLORS",
    "CROSSWALK_COL_MAP",
    "CROSSWALK_DTYPES",
    "EGRID_COL_MAP",
    "EGRID_DTYPES",
    "FOSSIL_INTENSITY",
    "FOSSIL_PRIME_MOVER_MAP",
    "FOSSIL_TECH",
    "GEN_COLS",
    "GEN_COLS_OLD",
    "GEN_TO_X_AGG",
    "MTDF",
    "PERMUTATIONS",
    "RAMP_CODES_TO_HRS",
    "RE_TECH",
    "ROOT_PATH",
    "SCALE_FACTOR_OVERRIDES",
    "STATE_BOUNDS",
]


ROOT_PATH = Path(__file__).parents[2]

"""
https://rockmtnins-my.sharepoint.com/:x:/g/personal/cfong_rmi_org/ES-9u5euInVFuO7KnpmjNiIByXnSx7gmAtBuNtET7-w6kQ
https://rockmtnins-my.sharepoint.com/:x:/g/personal/cfong_rmi_org/EcmLETNDwIlHjTk_YSEGiLYBz02VgHR1RG69rsKEAIFlkw
"""
MTDF = pd.DataFrame()
FOSSIL_PRIME_MOVER_MAP = {
    "ST": "ST",  # steam turbine
    "GT": "GT",  # gas turbine xCC
    "IC": "IC",  # internal combustion
    "CA": "CC",  # steam part of CC
    "CT": "CC",  # gas turbine part of CC
    "CS": "CC",  # single shaft CC
    "CC": "CC",  # CC in planning
}
FOSSIL_FUEL_CODES = [
    "ANT",  # Anthracite Coal
    "BIT",  # Bituminous Coal
    "LIG",  # Lignite Coal
    "SGC",  # Coal-Derived Synthesis Gas
    "SUB",  # Subbituminous Coal
    "WC",  # Waste/Other Coal (incl. anthracite culm, bituminous gob, fine coal, lignite waste, waste coal)
    "RC",  # Refined Coal
    "DFO",  # Distillate Fuel Oil (including diesel, No. 1, No. 2, and No. 4 fuel oils)
    "JF",  # Jet Fuel
    "KER",  # Kerosene
    "PC",  # Petroleum Coke
    "PG",  # Gaseous Propane
    "RFO",  # Residual Fuel Oil (incl. Nos. 5 & 6 fuel oils, and bunker C fuel oil)
    "SGP",  # Synthesis Gas from Petroleum Coke
    "WO",  # Waste/Other Oil (including crude oil, liquid butane, liquid propane, naphtha, oil waste, re-refined motor oil, sludge oil, tar oil, or other petroleum-based liquid wastes)
    "BFG",  # Blast Furnace Gas
    "NG",  # Natural Gas
    "OG",  # Other Gas (specify in SCHEDULE 7)
]
RAMP_CODES_TO_HRS = {
    "10M": 1,  # = 0 - 10 minutes
    "1H": 1,  # = 10 minutes - 1 hour
    "12H": 3,  # = 1 hour - 12 hours
    "OVER": 12,  # = More than 12 hours
}
"""
'Time from Cold Shutdown to Full Load' from EIA 860 3B
The minimum amount of time required to bring the unit to full load from shutdown
"""
CEMS_CARB_INTENSITY = {
    "BIT": 0.1027,
    "LIG": 0.10812,
    # "SGC": "coal",
    "SUB": 0.107065,
    # "WC": "coal",
    "RC": 0.125295,  # coke
    "DFO": 0.0817249,
    "JF": 0.079625,
    "KER": 0.080675,
    "PC": 0.112565,
    "PG": 0.069315,
    "RFO": 0.082775,
    # "SGP": "other_gas",
    # "WO": "petroleum",
    # "BFG": "other_gas",
    "NG": 0.058325,
    # "OG": "other_gas",
}
"""
https://www.eia.gov/environment/emissions/co2_vol_mass.php

in short_tons/mmbtu
"""

MIN_HEAT_RATE = {
    "Petroleum Liquids": 8.5,
    "Natural Gas Steam Turbine": 8.5,
    "Conventional Steam Coal": 8.5,
    "Natural Gas Fired Combined Cycle": 6.0,
    "Natural Gas Fired Combustion Turbine": 8.0,
    "Natural Gas Internal Combustion Engine": 7.0,
    "Coal Integrated Gasification Combined Cycle": 9.0,
    "Other Natural Gas": 8.5,
    "Petroleum Coke": 8.5,
    "Natural Gas with Compressed Air Storage": 8.5,
    "Other Gases": 8.5,
    "Wood/Wood Waste Biomass": 9.0,  # same as MSW
    "Municipal Solid Waste": 9.0,  # same as MSW
    "Landfill Gas": 9.0,  # same as MSW
    "All Other": 9.0,  # same as propane / other gases
    "Other Waste Biomass": 9.0,  # same as MSW
}
"""
in mmbtu/MWh
"""

FOSSIL_INTENSITY = {
    "coal": 9.61825e-08,
    "natural_gas": 5.306e-08,
    "gas": 5.306e-08,
    "other_gas": 1.46972e-07,
    "petroleum_coke": 1.0241e-07,
    "petroleum": 7.18297e-08,
    "oil": 7.18297e-08,
}
"""Unknown units"""

CEMS_COL_MAP = {
    "ORISPL_CODE": "plant_id_cems",
    "UNITID": "unit_id_cems",
    "GLOAD..MW.": "gross_gen",
    "HEAT_INPUT..mmBtu.": "heat_in_mmbtu",
    "CO2_MASS..tons.": "co2_tons",
    "OP_TIME": "op_time",
}
CROSSWALK_DTYPES = {
    "CAMD_PLANT_ID": "Int64",
    "CAMD_UNIT_ID": str,
    "CAMD_GENERATOR_ID": str,
    "EIA_PLANT_ID": "Int64",
    "EIA_GENERATOR_ID": str,
    "EIA_BOILER_ID": str,
    "MOD_EIA_PLANT_ID": "Int64",
}
CROSSWALK_COL_MAP = {
    "EIA_PLANT_ID": "plant_id_eia",
    "EIA_GENERATOR_ID": "generator_id",
    "EIA_BOILER_ID": "boiler_id",
    "EIA_STATE": "state",
    "EIA_UNIT_TYPE": "prime_mover_code",
    "EIA_FUEL_TYPE": "energy_source_code_1",
    "EIA_NAMEPLATE_CAPACITY": "capacity_mw",
    "EIA_PLANT_NAME": "plant_name_eia",
    "CAMD_PLANT_ID": "plant_id_cems",
    "CAMD_UNIT_ID": "unit_id_cems",
    "CAMD_GENERATOR_ID": "generator_id_cems",
    "CAMD_FUEL_TYPE": "fuel_code_cems",
    "CAMD_NAMEPLATE_CAPACITY": "capacity_mw_cems",
    "CAMD_FACILITY_NAME": "plant_name_cems",
    "MOD_EIA_PLANT_ID": "plant_id_eia_mod",
    "MOD_EIA_BOILER_ID": "boiler_id_eia_mod",
    "MOD_EIA_GENERATOR_ID_BOILER": "generator_id_boiler_eia_mod",
    "MOD_EIA_GENERATOR_ID_GEN": "generator_id_gen_eia_mod",
    "MATCH_TYPE_GEN": "match_type_gen",
    "MATCH_TYPE_BOILER": "match_type_boiler",
    "PLANT_ID_CHANGE_FLAG": "plant_id_change",
}
EGRID_DTYPES = {
    "ORISPL": "Int64",
    "UNITID": str,
    "PRMVR": str,
    "UNTOPST": str,
    "FUELU1": str,
    "HTIANSRC": str,
}
EGRID_COL_MAP = {
    "PSTATABB": "state",
    "PNAME": "plant_name_cems",
    "ORISPL": "plant_id_cems",
    "UNITID": "unit_id_cems",
    "PRMVR": "prime_mover_code",
    "UNTOPST": "op_status",
    "FUELU1": "energy_source_code_1",
    "HTIANSRC": "heat_input_source",
}
GEN_TO_X_AGG = {
    "generator_id": set,
    "state": "first",
    "report_date": "first",
    "plant_name_eia": "first",
    "utility_id_eia": "first",
    "utility_name_eia": "first",
    "technology_description": set,
    "capacity_mw": np.sum,
    "latitude": "first",
    "longitude": "first",
    "operating_date": "first",
    "operational_status": set,
    # 'original_planned_operating_date',
    "timezone": "first",
    "balancing_authority_code_eia": "first",
    # 'balancing_authority_name_eia',
    # 'cofire_fuels',
    "fuel_type_code_pudl": set,
    "energy_source_code_1": set,
    "plant_id_pudl": "first",
    "utility_id_pudl": "first",
}
# TODO reordering to (off, on, solar) would probably solve most order driven weirdness
PERMUTATIONS = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
        [0.0, 0.2, 0.8],
        [0.0, 0.3, 0.7],
        [0.0, 0.4, 0.6],
        [0.0, 0.6, 0.4],
        [0.0, 0.7, 0.3],
        [0.0, 0.8, 0.2],
        [0.0, 0.9, 0.1],
        [0.0, 1.0, 0.0],
        [0.1, 0.0, 0.9],
        [0.1, 0.2, 0.7],
        [0.1, 0.3, 0.6],
        [0.1, 0.4, 0.5],
        [0.1, 0.5, 0.4],
        [0.1, 0.6, 0.3],
        [0.1, 0.7, 0.2],
        [0.1, 0.9, 0.0],
        [0.2, 0.0, 0.8],
        [0.2, 0.1, 0.7],
        [0.2, 0.3, 0.5],
        [0.2, 0.5, 0.3],
        [0.2, 0.7, 0.1],
        [0.2, 0.8, 0.0],
        [0.3, 0.0, 0.7],
        [0.3, 0.1, 0.6],
        [0.3, 0.2, 0.5],
        [0.3, 0.5, 0.2],
        [0.3, 0.6, 0.1],
        [0.3, 0.7, 0.0],
        [0.4, 0.0, 0.6],
        [0.4, 0.1, 0.5],
        [0.4, 0.5, 0.1],
        [0.4, 0.6, 0.0],
        [0.5, 0.1, 0.4],
        [0.5, 0.2, 0.3],
        [0.5, 0.3, 0.2],
        [0.5, 0.4, 0.1],
        [0.6, 0.0, 0.4],
        [0.6, 0.1, 0.3],
        [0.6, 0.3, 0.1],
        [0.6, 0.4, 0.0],
        [0.7, 0.0, 0.3],
        [0.7, 0.1, 0.2],
        [0.7, 0.2, 0.1],
        [0.7, 0.3, 0.0],
        [0.8, 0.0, 0.2],
        [0.8, 0.2, 0.0],
        [0.9, 0.0, 0.1],
        [0.9, 0.1, 0.0],
        [1.0, 0.0, 0.0],
    ]
)
STATE_BOUNDS = {
    "xmin": {
        "AL": -88.473227,
        "AK": -179.148909,
        "AS": -171.089874,
        "AZ": -114.81651,
        "AR": -94.617919,
        "CA": -124.409591,
        "CO": -109.060253,
        "MP": 144.886331,
        "CT": -73.727775,
        "DE": -75.788658,
        "DC": -77.119759,
        "FL": -87.634938,
        "GA": -85.605165,
        "GU": 144.618068,
        "HI": -178.334698,
        "ID": -117.243027,
        "IL": -91.513079,
        "IN": -88.09776,
        "IA": -96.639704,
        "KS": -102.051744,
        "KY": -89.571509,
        "LA": -94.043147,
        "ME": -71.083924,
        "MD": -79.487651,
        "MA": -73.508142,
        "MI": -90.418136,
        "MN": -97.239209,
        "MS": -91.655009,
        "MO": -95.774704,
        "MT": -116.050003,
        "NE": -104.053514,
        "NV": -120.005746,
        "NH": -72.557247,
        "NJ": -75.559614,
        "NM": -109.050173,
        "NY": -79.762152,
        "NC": -84.321869,
        "ND": -104.0489,
        "OH": -84.820159,
        "OK": -103.002565,
        "OR": -124.566244,
        "PA": -80.519891,
        "PR": -67.945404,
        "RI": -71.862772,
        "SC": -83.35391,
        "SD": -104.057698,
        "TN": -90.310298,
        "TX": -106.645646,
        "VI": -65.085452,
        "UT": -114.052962,
        "VT": -73.43774,
        "VA": -83.675395,
        "WA": -124.763068,
        "WV": -82.644739,
        "WI": -92.888114,
        "WY": -111.056888,
    },
    "xmax": {
        "AL": -84.88908,
        "AK": 179.77847,
        "AS": -168.1433,
        "AZ": -109.045223,
        "AR": -89.644395,
        "CA": -114.131211,
        "CO": -102.041524,
        "MP": 146.064818,
        "CT": -71.786994,
        "DE": -75.048939,
        "DC": -76.909395,
        "FL": -80.031362,
        "GA": -80.839729,
        "GU": 144.956712,
        "HI": -154.806773,
        "ID": -111.043564,
        "IL": -87.494756,
        "IN": -84.784579,
        "IA": -90.140061,
        "KS": -94.588413,
        "KY": -81.964971,
        "LA": -88.817017,
        "ME": -66.949895,
        "MD": -75.048939,
        "MA": -69.928393,
        "MI": -82.413474,
        "MN": -89.491739,
        "MS": -88.097888,
        "MO": -89.098843,
        "MT": -104.039138,
        "NE": -95.30829,
        "NV": -114.039648,
        "NH": -70.610621,
        "NJ": -73.893979,
        "NM": -103.001964,
        "NY": -71.856214,
        "NC": -75.460621,
        "ND": -96.554507,
        "OH": -80.518693,
        "OK": -94.430662,
        "OR": -116.463504,
        "PA": -74.689516,
        "PR": -65.220703,
        "RI": -71.12057,
        "SC": -78.54203,
        "SD": -96.436589,
        "TN": -81.6469,
        "TX": -93.508292,
        "VI": -64.564907,
        "UT": -109.041058,
        "VT": -71.464555,
        "VA": -75.242266,
        "WA": -116.915989,
        "WV": -77.719519,
        "WI": -86.805415,
        "WY": -104.05216,
    },
    "ymin": {
        "AL": 30.223334,
        "AK": 51.214183,
        "AS": -14.548699,
        "AZ": 31.332177,
        "AR": 33.004106,
        "CA": 32.534156,
        "CO": 36.992426,
        "MP": 14.110472,
        "CT": 40.980144,
        "DE": 38.451013,
        "DC": 38.791645,
        "FL": 24.523096,
        "GA": 30.357851,
        "GU": 13.234189,
        "HI": 18.910361,
        "ID": 41.988057,
        "IL": 36.970298,
        "IN": 37.771742,
        "IA": 40.375501,
        "KS": 36.993016,
        "KY": 36.497129,
        "LA": 28.928609,
        "ME": 42.977764,
        "MD": 37.911717,
        "MA": 41.237964,
        "MI": 41.696118,
        "MN": 43.499356,
        "MS": 30.173943,
        "MO": 35.995683,
        "MT": 44.358221,
        "NE": 39.999998,
        "NV": 35.001857,
        "NH": 42.69699,
        "NJ": 38.928519,
        "NM": 31.332301,
        "NY": 40.496103,
        "NC": 33.842316,
        "ND": 45.935054,
        "OH": 38.403202,
        "OK": 33.615833,
        "OR": 41.991794,
        "PA": 39.7198,
        "PR": 17.88328,
        "RI": 41.146339,
        "SC": 32.0346,
        "SD": 42.479635,
        "TN": 34.982972,
        "TX": 25.837377,
        "VI": 17.673976,
        "UT": 36.997968,
        "VT": 42.726853,
        "VA": 36.540738,
        "WA": 45.543541,
        "WV": 37.201483,
        "WI": 42.491983,
        "WY": 40.994746,
    },
    "ymax": {
        "AL": 35.008028,
        "AK": 71.365162,
        "AS": -11.046934,
        "AZ": 37.00426,
        "AR": 36.4996,
        "CA": 42.009518,
        "CO": 41.003444,
        "MP": 20.553802,
        "CT": 42.050587,
        "DE": 39.839007,
        "DC": 38.99511,
        "FL": 31.000888,
        "GA": 35.000659,
        "GU": 13.654383,
        "HI": 28.402123,
        "ID": 49.001146,
        "IL": 42.508481,
        "IN": 41.760592,
        "IA": 43.501196,
        "KS": 40.003162,
        "KY": 39.147458,
        "LA": 33.019457,
        "ME": 47.459686,
        "MD": 39.723043,
        "MA": 42.886589,
        "MI": 48.2388,
        "MN": 49.384358,
        "MS": 34.996052,
        "MO": 40.61364,
        "MT": 49.00139,
        "NE": 43.001708,
        "NV": 42.002207,
        "NH": 45.305476,
        "NJ": 41.357423,
        "NM": 37.000232,
        "NY": 45.01585,
        "NC": 36.588117,
        "ND": 49.000574,
        "OH": 41.977523,
        "OK": 37.002206,
        "OR": 46.292035,
        "PA": 42.26986,
        "PR": 18.515683,
        "RI": 42.018798,
        "SC": 35.215402,
        "SD": 45.94545,
        "TN": 36.678118,
        "TX": 36.500704,
        "VI": 18.412655,
        "UT": 42.001567,
        "VT": 45.016659,
        "VA": 39.466012,
        "WA": 49.002494,
        "WV": 40.638801,
        "WI": 47.080621,
        "WY": 45.005904,
    },
}
FOSSIL_TECH = [
    "Petroleum Liquids",
    "Natural Gas Steam Turbine",
    "Conventional Steam Coal",
    "Natural Gas Fired Combined Cycle",
    "Natural Gas Fired Combustion Turbine",
    "Natural Gas Internal Combustion Engine",
    "Coal Integrated Gasification Combined Cycle",
    "Other Natural Gas",
    "Petroleum Coke",
    "Natural Gas with Compressed Air Storage",
    "Other Gases",
]
RE_TECH = {
    "Solar Photovoltaic": "solar",
    "Onshore Wind Turbine": "onshore_wind",
    "Offshore Wind Turbine": "offshore_wind",
    "Solar": "solar",
    "Wind": "onshore_wind",
    "Offshore_Wind": "offshore_wind",
}
RE_TECH_R = {
    "solar": "Solar Photovoltaic",
    "onshore_wind": "Onshore Wind Turbine",
    "offshore_wind": "Offshore Wind Turbine",
}
GEN_COLS = [
    "plant_id_eia",
    "plant_name_eia",
    "utility_id_eia",
    # "utility_name_eia",
    "generator_id",
    "state",
    "final_ba_code",
    "final_respondent_id",
    "respondent_name",
    "balancing_authority_code_eia",
    "capacity_mw",
    "prime_mover_code",
    "prime_mover",
    "technology_description",
    "energy_source_code_860m",
    "fuel_group_energy_source_code_860m",
    "rmi_energy_source_code_1",
    "rmi_energy_source_code_2",
    "rmi_energy_source_code_3",
    "fuel_group_rmi_energy_source_code_1",
    "fuel_group_rmi_energy_source_code_2",
    "fuel_group_rmi_energy_source_code_3",
    "cofire_fuels",
    "multiple_fuels",
    "status_860m",
    "operational_status",
    # "original_planned_operating_date",
    # "timezone",
    # "ramp_up_rate",
    "ramp_hrs",
    "latitude",
    "longitude",
    "operating_date",
    "retirement_date",
    # "operating_month",
    # "operating_year",
    # "retirement_month",
    # "retirement_year",
    # "co2_factor",
]
GEN_COLS_OLD = [
    "report_date",
    "plant_id_eia",
    "plant_name_eia",
    "utility_id_eia",
    "utility_name_eia",
    "generator_id",
    "technology_description",
    "capacity_mw",
    "latitude",
    "longitude",
    "operating_date",
    "operational_status",
    "original_planned_operating_date",
    "prime_mover_code",
    "timezone",
    "balancing_authority_code_eia",
    "balancing_authority_name_eia",
    "cofire_fuels",
    "fuel_type_code_pudl",
    "energy_source_code_1",
    "energy_source_code_2",
    "energy_source_code_3",
    "plant_id_pudl",
    "utility_id_pudl",
    "state",
]
COLORS = {
    "solar": "#FFCB00",
    "onshore_wind": "#3DADF2",
    "offshore_wind": "#0066B3",
    "fossil": "#E04D39",
}
SCALE_FACTOR_OVERRIDES = pd.DataFrame(
    columns=[
        "plant_id_eia",
        "prime_mover_code",
        "fuel_code",
        "report_date",
        "capacity",
    ],
    data=[
        # these started producing energy before 860 online date
        (1378, "CC", "natural_gas", 2016, 1160.0),  # Paradise
        (6190, "ST", "petroleum_coke", 2009, 703.8),  # Brame Energy Center
        (56846, "CC", "natural_gas", 2016, 775.3),  # CPV St Charles Energy Center
        (57794, "CC", "natural_gas", 2017, 780.0),  # St Joseph Energy Center
        (59220, "CC", "natural_gas", 2017, 1113.6),  # Wildcat Point Generation Facility
        (60264, "GT", "natural_gas", 2017, 427.2),  # Bacliff
        (61241, "GT", "natural_gas", 2019, 100.0),  # Victoria City Power LLC
        # Deerhaven
        (663, "ST", "coal", 2008, 250.7),
        (663, "ST", "coal", 2009, 250.7),
        (663, "ST", "coal", 2010, 250.7),
        (663, "ST", "coal", 2011, 250.7),
        (663, "ST", "coal", 2012, 250.7),
        (663, "ST", "coal", 2013, 250.7),
        (663, "ST", "coal", 2014, 250.7),
        (663, "ST", "coal", 2015, 250.7),
        (663, "ST", "coal", 2016, 250.7),
        (663, "ST", "coal", 2017, 250.7),
        (663, "ST", "coal", 2018, 250.7),
        (663, "ST", "coal", 2019, 250.7),
        (663, "ST", "coal", 2020, 250.7),
        # (663, 'ST', 'coal', 2021, 250.7),
        # Chalk Point (retired 2021)
        (1571, "ST", "coal", 2008, 728.0),
        (1571, "ST", "coal", 2009, 728.0),
        (1571, "ST", "coal", 2010, 728.0),
        (1571, "ST", "coal", 2011, 728.0),
        (1571, "ST", "coal", 2012, 728.0),
        (1571, "ST", "coal", 2013, 728.0),
        (1571, "ST", "coal", 2014, 728.0),
        (1571, "ST", "coal", 2015, 728.0),
        (1571, "ST", "coal", 2016, 728.0),
        (1571, "ST", "coal", 2017, 728.0),
        (1571, "ST", "coal", 2018, 728.0),
        (1571, "ST", "coal", 2019, 728.0),
        (1571, "ST", "coal", 2020, 728.0),
        # (1571, 'ST', 'coal', 2021, 0),
    ],
    index=range(33),
).astype(
    {
        "plant_id_eia": int,
        "prime_mover_code": str,
        "fuel_code": str,
        "report_date": int,
        "capacity": float,
    }
)

BAD_RE = (
    #     "56540_solar",
    #     "62071_solar",
    #     "6304_solar",
    #     "63169_solar",
    #     "63951_solar",
    #     "64122_solar",
    #     "64159_solar",
    #     "64462_solar",
    #     "64827_solar",
    #     "64905_solar",
    #     "64971_solar",
    #     "65013_solar",
    #     "65349_solar",
    #     "65417_solar",
    #     "65508_solar",
    #     "65726_solar",
    #     "65770_solar",
    #     "65777_solar",
    #     "9990400_onshore_wind",
    #     "9990402_onshore_wind",
    #     "9990448_onshore_wind",
    #     "9990482_onshore_wind",
    #     "9990493_onshore_wind",
    #     "9990585_onshore_wind",
    #     "9990972_onshore_wind",
)
PATIO_DOC_PATH = Path.home() / "patio_data"
ES_TECHS = ("Batteries", "Hydroelectric Pumped Storage", "H2 Storage")
REGION_MAP = {
    "WALC": "West",
    "AECI": "Southeast",
    "EPE": "West",
    "LDWP": "West",
    "PJM": "PJM",
    "LGEE": "Southeast",
    "22": "MISO",
    "PAC": "West",
    "PNM": "West",
    "SWPP": "SPP",
    "CAISO": "CAISO",
    "195": "MISO",
    "EVRG": "SPP",
    "658": "SPP",
    "AEC": "Southeast",
    "569": "Southeast",
    "SCEG": "Southeast",
    "TEC": "Southeast",
    "58": "SPP",
    "FPC": "Southeast",
    "NYIS": "NYISO",
    "NEVP": "West",
    "WACM": "West",
    #  'NBSO': 'NBSO',
    #  'HECO': 'HECO',
    "SC": "Southeast",
    "552": "MISO",
    "PSCO": "West",
    "TEPC": "West",
    "TVA": "Southeast",
    "Alaska": "West",
    "DUKE": "Southeast",
    "MISO": "MISO",
    "ETR": "MISO",
    "210": "MISO",
    "SOCO": "Southeast",
    "JEA": "Southeast",
    "SRP": "West",
    "ISNE": "ISONE",
    "WAUW": "West",
    "177": "MISO",
    "556": "PJM",
    "656": "SPP",
    "2": "Southeast",
    "FPL": "Southeast",
    "FMPP": "Southeast",
    "531": "SPP",
    "186": "PJM",
    "130": "SPP",
    "APS": "West",
    "SEC": "Southeast",
    "57": "Southeast",
    "ERCO": "ERCOT",
    "PNW": "West",
}
CCS_FACTORS = pd.Series(
    {
        "vom_per_mwh": 14.75,  # CCS VOM
        "fom_per_kw": 130.0,  # CC FOM
        # heat rate factor but named this way so pandas does the multiplication right
        "fuel_per_mwh": 1.27,  # Heat Rate Penalty (Δ% from pre-retrofit)
        "heat_rate": 1.27,  # Heat Rate Penalty (Δ% from pre-retrofit)
        "co2_factor": 0.1,  # 90% capture
    }
)
"""CCS adders/factors.
NREL ATB 2023, Coal-new vs Coal integrated retrofit 90%-CCS, 2035 moderate case.
"""
CARB_INTENSITY = {
    "Petroleum Liquids": 0.07509,
    "Natural Gas Steam Turbine": 0.05291,
    "Conventional Steam Coal": 0.09610,
    "Natural Gas Fired Combined Cycle": 0.05291,
    "Natural Gas Fired Combustion Turbine": 0.05291,
    "Natural Gas Internal Combustion Engine": 0.05291,
    "Coal Integrated Gasification Combined Cycle": 0.09610,
    "Other Natural Gas": 0.05291,
    "Petroleum Coke": 0.10212,
    "Natural Gas with Compressed Air Storage": 0.05291,
    "Other Gases": 0.06288,
    "Solar Photovoltaic": 0.0,
    "Onshore Wind Turbine": 0.0,
    "Offshore Wind Turbine": 0.0,
    "Solar": 0.0,
    "Wind": 0.0,
    "Offshore_Wind": 0.0,
    "Wood/Wood Waste Biomass": 0.04989,  # same as MSW
    "Municipal Solid Waste": 0.04989,  # same as MSW
    "Landfill Gas": 0.04989,  # same as MSW
    "All Other": 0.06288,  # same as propane / other gases
    "Other Waste Biomass": 0.04989,  # same as MSW
    "Nuclear": 0.0,
    "Geothermal": 0.0,
}
COST_FLOOR = pd.DataFrame(
    [
        ("Coal Integrated Gasification Combined Cycle", 5.0, 35.0, 0.1, 10.0),
        ("Conventional Steam Coal", 4.0, 25.0, 0.1, 10.0),
        ("Landfill Gas", 4.0, 15.0, 0.1, 10.0),
        ("Municipal Solid Waste", 4.0, 15.0, 0.1, 10.0),
        ("Natural Gas Fired Combined Cycle", 2.0, 20.0, 0.1, 15.0),
        ("Natural Gas Fired Combustion Turbine", 5.0, 15.0, 0.1, 20.0),
        ("Natural Gas Internal Combustion Engine", 4.0, 15.0, 0.1, 15.0),
        ("Natural Gas Steam Turbine", 4.0, 20.0, 0.1, 20.0),
        ("Other Gases", 5.0, 15.0, 0.1, 10.0),
        ("Other Waste Biomass", 4.0, 15.0, 0.1, 10.0),
        ("Petroleum Coke", 5.0, 15.0, 0.1, 30.0),
        ("Petroleum Liquids", 5.0, 15.0, 0.1, 40.0),
        ("Wood/Wood Waste Biomass", 4.0, 15.0, 0.1, 10.0),
        # actual floor not needed because these costs come from ATB
        ("Solar Photovoltaic", 0.0, 0.0, 0.0, 0.0),
        ("Onshore Wind Turbine", 0.0, 0.0, 0.0, 0.0),
        ("Offshore Wind Turbine", 0.0, 0.0, 0.0, 0.0),
        ("Geothermal", 0.0, 0.0, 0.0, 0.0),
        ("Nuclear", 0.0, 0.0, 0.0, 0.0),
    ],
    columns=[
        "technology_description",
        "vom_floor",
        "fom_floor",
        "som_floor",
        "fuel_floor",
    ],
)
UDAY_FOSSIL_FUEL_MAP = {  # the better map from Uday
    "ANT": "coal",
    "BIT": "coal",
    "LIG": "coal",
    "SGC": "coal",
    "SUB": "coal",
    "WC": "coal",
    "RC": "coal",
    "DFO": "petroleum",
    "JF": "petroleum",
    "KER": "petroleum",
    "PC": "petroleum_coke",
    "PG": "petroleum",
    "RFO": "petroleum",
    "SGP": "other_gas",
    "WO": "petroleum",
    "BFG": "other_gas",
    "NG": "natural_gas",
    "OG": "other_gas",
    "SC": "coal",
    "TDF": "petroleum",
}
FUEL_GROUP_MAP = UDAY_FOSSIL_FUEL_MAP | {
    "AB": "biofuel",  # other in fuel_group_emissions_map
    "MSW": "other",  # other in fuel_group_emissions_map
    "OBS": "biofuel",  # other in fuel_group_emissions_map
    "WDS": "biofuel",  # other in fuel_group_emissions_map
    "OBL": "biofuel",  # other in fuel_group_emissions_map
    "SLW": "other",  # other in fuel_group_emissions_map
    "BLQ": "biofuel",  # other in fuel_group_emissions_map
    "WDL": "biofuel",  # other in fuel_group_emissions_map
    "LFG": "biofuel",  # other in fuel_group_emissions_map
    "OBG": "biofuel",  # other in fuel_group_emissions_map
    "MSB": "biofuel",  # other in fuel_group_emissions_map
    "SUN": "renew",  # other in fuel_group_emissions_map
    "WND": "renew",  # other in fuel_group_emissions_map
    "GEO": "renew",  # other in fuel_group_emissions_map
    "WAT": "renew",  # other in fuel_group_emissions_map
    "NUC": "nuclear",  # other in fuel_group_emissions_map
    "WH": "other",
    "MWH": "other",
    "OTH": "other",
    "MSN": "other",
    "PUR": "other",
}
CLEAN_TD_MAP = {
    "Solar Photovoltaic": "solar",
    "Onshore Wind Turbine": "onshore_wind",
    "Offshore Wind Turbine": "offshore_wind",
    "Nuclear": "nuclear",
}
OTHER_TD_MAP = {
    "Conventional Steam Coal": "coal",
    "Coal Integrated Gasification Combined Cycle": "coal",
    "Natural Gas Fired Combined Cycle": "natural_gas_cc",
    "Natural Gas Fired Combustion Turbine": "natural_gas_gt",
    "Natural Gas Internal Combustion Engine": "natural_gas_ic",
    "Natural Gas Steam Turbine": "natural_gas_st",
    "Other Natural Gas": "natural_gas_ot",
    "Petroleum Liquids": "petroleum",
    "Other Gases": "other",
    "Petroleum Coke": "petroleum",
    "Wood/Wood Waste Biomass": "biomass",
    "Municipal Solid Waste": "biomass",
    "Landfill Gas": "biomass",
    "Other Waste Biomass": "biomass",
    "All Other": "other",
}
TD_MAP = CLEAN_TD_MAP | OTHER_TD_MAP
SIMPLE_TD_MAP = CLEAN_TD_MAP | {
    "Conventional Steam Coal": "coal",
    "Coal Integrated Gasification Combined Cycle": "coal",
    "Natural Gas Fired Combined Cycle": "natural_gas_cc",
    "Natural Gas Fired Combustion Turbine": "natural_gas_gt",
    "Natural Gas Internal Combustion Engine": "natural_gas_ot",
    "Natural Gas Steam Turbine": "natural_gas_ot",
    "Other Natural Gas": "natural_gas_ot",
    "Petroleum Liquids": "other",
    "Other Gases": "other",
    "Petroleum Coke": "other",
    "Wood/Wood Waste Biomass": "biomass",
    "Municipal Solid Waste": "biomass",
    "Landfill Gas": "biomass",
    "Other Waste Biomass": "biomass",
    "All Other": "other",
}
PATIO_PUDL_RELEASE = "v2024.11.0"
PATIO_DATA_RELEASE = "20241031"
PATIO_DATA_AZURE_URLS = {
    # "ba_cems": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/ba_cems.zip",
    # "ba_cems_extended": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/ba_cems_extended.zip",
    "ba_cems": f"patio-data/{PATIO_DATA_RELEASE}/ba_cems.zip",
    "ba_cems_extended": f"patio-data/{PATIO_DATA_RELEASE}/ba_cems_extended.zip",
    "camd_starts_ms": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/camd_starts_ms.parquet",
    "Coal_Closure_Energy_Communities_SHP_2023v2": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/Coal_Closure_Energy_Communities_SHP_2023v2.zip",
    "irp": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/irp.parquet",
    "MSA_NMSA_FEE_EC_Status_SHP_2023v2": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/MSA_NMSA_FEE_EC_Status_SHP_2023v2.zip",
    "re_data_limited": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/re_data_limited.zip",
    "re_data_reference": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/re_data_reference.zip",
    # "re_data": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/re_data.zip",
    "re_data": f"patio-data/{PATIO_DATA_RELEASE}/re_data.zip",
    "lse": f"https://rmicfezil.blob.core.windows.net/patio-data/{PATIO_DATA_RELEASE}/Electric_Retail_Service_Territories.kml.zip",
}
TECH_CODES = {
    "coal": "Conventional Steam Coal",
    "ngcc": "Natural Gas Fired Combined Cycle",
    "ngct": "Natural Gas Fired Combustion Turbine",
    "nggt": "Natural Gas Fired Combustion Turbine",
    "ngic": "Natural Gas Internal Combustion Engine",
    "ngst": "Natural Gas Steam Turbine",
    "ngot": "Other Natural Gas",
    "petroleum": "Petroleum Coke",
    "other": "All Other",
    "nuke": "Nuclear",
    "nuclear": "Nuclear",
    "solar": "Solar Photovoltaic",
    "wind": "Onshore Wind Turbine",
    "onshore_wind": "Onshore Wind Turbine",
    "offshore_wind": "Offshore Wind Turbine",
}
