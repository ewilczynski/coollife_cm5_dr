# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:15:40 2024

@author: ewilczynski
"""
import pandas as pd
import numpy as np
import pvlib
import requests
import os
import datetime as dt
#import time
#import math

import logging
from pathlib import Path
from typing import Dict, Tuple
#from BaseCM.cm_output import validate
#from shapely.geometry import Polygon, Point, MultiPolygon
#import shapefile

import pyomo.environ as en
from pyomo.core import Var
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
from pyomo.util.infeasible import log_infeasible_constraints
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


DECIMALS = 3
CURRENT_FILE_DIR = Path(__file__).parent
TESTDATA_DIR = CURRENT_FILE_DIR / "Input"

session_dict = {}
param_dict = {}
session_dict["roof_tilt"] = 30
session_dict["roof_azimuth"] = 180

def load_parameters(country_name):
    # Specify time range for typical cooling week
    session_dict["month_number"] = 8
    session_dict["start_day"] = 4 #note, actual start will be 24 hours later DATE-X
    session_dict["start"] = pd.Timestamp('2020-08-' + str(session_dict["start_day"]))#, tz='utc') #DATE-X
    session_dict["end_day"] = 6 #DATE-X
    session_dict["end"] = pd.Timestamp('2020-08-' + str(session_dict["end_day"]))#, tz='utc') #DATE-X
    session_dict["year"] = 2020 #DATE-X
    session_dict["roof_type_orientation"] = 4
    session_dict["n_stories"] = 1
    session_dict["scenario"] = "PV Match" #"PV Match" or "Peak Shaving"
    session_dict["design_choice"] = "Self-sufficiency" #choice between "Balanced", "Low CO2", "Self-sufficiency"
    session_dict["conditioning"] = "Cooling" # or "Heating"
    session_dict["ubound"] = 26
    session_dict["lbound"] = 20
    
    #Temporary:
    eu27_centers = {
        'AT': (47.5162, 14.5501),  # Austria
        'BE': (50.8503, 4.3517),   # Belgium
        'BG': (42.7339, 25.4858),  # Bulgaria
        'HR': (45.1000, 15.2000),  # Croatia
        'CY': (35.1264, 33.4299),  # Cyprus
        'CZ': (49.8175, 15.4730),  # Czech Republic
        'DK': (56.2639, 9.5018),   # Denmark
        'EE': (58.5953, 25.0136),  # Estonia
        'FI': (61.9241, 25.7482),  # Finland
        'FR': (46.6034, 1.8883),   # France
        'DE': (51.1657, 10.4515),  # Germany
        'EL': (39.0742, 21.8243),  # Greece
        'HU': (47.1625, 19.5033),  # Hungary
        'IE': (53.1424, -7.6921),  # Ireland
        'IT': (41.8719, 12.5674),  # Italy
        'LV': (56.8796, 24.6032),  # Latvia
        'LT': (55.1694, 23.8813),  # Lithuania
        'LU': (49.8153, 6.1296),   # Luxembourg
        'MT': (35.9375, 14.3754),  # Malta
        'NL': (52.1326, 5.2913),   # Netherlands
        'PL': (51.9194, 19.1451),  # Poland
        'PT': (39.3999, -8.2245),  # Portugal
        'RO': (45.9432, 24.9668),  # Romania
        'SK': (48.6690, 19.6990),  # Slovakia
        'SI': (46.1512, 14.9955),  # Slovenia
        'ES': (40.4637, -3.7492),  # Spain
        'SE': (60.1282, 18.6435)   # Sweden
    }
    session_dict["country_name"] = country_name
    country_name__bycountrycode = {'AT':'Austria', 'BE':'Belgium', 'BG':'Bulgaria', 'HR':'Croatia', 'CY':'Cyprus', 'CZ':'Czech Republic', 'DE':'Germany', 'DK':'Denmark', 'EE':'Estonia', 'ES':'Spain', 'FR':'France', 'FI':'Finland', 'GR':'Greece', 'HU':'Hungary', 'IE':'Ireland', 'IT':'Italy', 'LV':'Latvia','LT':'Lithuania', 'LU':'Luxembourg', 'MT':'Malta', 'NL':'Netherlands', 'NO':'Norway', 'PL':'Poland', 'PT':'Portugal', 'RO':'Romania', 'SE':'Sweden', 'SK':'Slovakia', 'SI':'Slovenia', 'UK':'United Kingdom'}    
    session_dict["country_code"] = list(country_name__bycountrycode.keys())[list(country_name__bycountrycode.values()).index(country_name)]
    session_dict["lat"] = eu27_centers[session_dict["country_code"]][0]
    session_dict["long"] = eu27_centers[session_dict["country_code"]][1]
    return session_dict

#------------------------------------------------------------------------------
# Parameter data
#------------------------------------------------------------------------------
pm = {
        # Parameter for calculating effective mass area, A_m (Table 12)
        "A_m_parameter": 2.5,
        # Parameter for internal heat capacity, C_m (Table 12)
        "C_m_parameter": 45 * 3600, 
        # Reduction factor external shading, horizontal orientation (from Tabula)
        "F_sh_hor": 0.8,
        # Reduction factor external shading, vertical orientations (from Tabula)
        "F_sh_vert": 0.6,
        # Form factor for radiation between the element and the sky for unshaded horizontal roof
        "F_r_hor": 1.0,
        # Form factor for radiation between the element and the sky for unshaded vertical roof
        "F_r_vert": 0.5,      
        # Standard room height (m) (from Tabula)
        "h_room": 2.5,
        # Heat transfer coefficient between the air node and surface node (W/(m²K))
        "h_is": 3.45,
        # Dimensionless ratio between the internal surfaces area and the floor area
        "lambda_at": 4.5,
        # Heat transfer coefficient between nodes m and s
        "h_ms": 9.1,
        # Heat capacity of air per volume (J/(m^3 K))
        "rho_a_c_a": 1200,
        # Average heat flow per person (W/person)
        "Q_P": 70,
        # Frame area fraction
        "F_F": 0.3,
        # Frame area fraction
        "F_w": 0.9,
        # Total solar energy transmittance for radiation perpendicular to the glazing (from Tabula)
        "g_gl_n": 0.6,
        # Air change rate from infiltration (per hour) (Tabula value for "medium")
        "n_air_inf": 0.2,
        # Air change rate from use (per hour) (from Tabula)
        "n_air_use": 0.4,
        # Dimensionless absorption coefficient for solar radiation of the opaque part
        "alpha_S_c": 0.6,
        # Average difference between the external air temperature and the apparent sky temperature (degC)
        "delta_theta_er": 11,
        # Arithmetic average of the surface temperature and the sky temperature (degC)
        "theta_ss": 11,
        # Stefan-Boltzmann constant (W/(m^2 * K^4))
        "SB_constant": 5.67 * (10**-8),
        # Emissivity of external opaque elements (roof, wall)
        "epsilon_op": 0.9,
        # Emissivity of external glazed elements (window, door)
        "epsilon_gl": 0.9,
        # External surface heat resistance of the opaque part (ISO 6946)
        "R_S_e_op": 0.04,
         # External surface heat resistance of the glazed part (ISO 6946)
        "R_S_e_gl": 0.04,      
        # Parameter for calculation of conditioned floor area (from Tabula calculations)
        "gfa_parameter": 0.85,
        }

#------------------------------------------------------------------------------
# 1a. Transform user input to coordinates
#------------------------------------------------------------------------------
def compute_centroid(geojson: Dict) -> Tuple[float, float]:
    try:
        coords = np.array(geojson["features"][0]["geometry"]["coordinates"])
    except KeyError:
        logging.error(geojson)
        raise ValueError(
            "FAILED! The provided geometry is not a correct/supported geojson format."
        )
    return tuple(np.around(coords[0].mean(0), decimals=DECIMALS))

#TODO uncomment for prod
#session_dict["long"], session_dict["lat"] = compute_centroid(geojson)

#------------------------------------------------------------------------------
# 1b. Get country code
#------------------------------------------------------------------------------
def countrycode(
        #geojson: Dict,
        #long: float,
        #lat: float
        ):
    # Read shapefile
    path = 'Input/countries.shp'
    sf = shapefile.Reader(path)
    
    lat = session_dict["lat"]
    long = session_dict["long"]
    
    country_list = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Germany', 'Denmark', 'Estonia', 'Spain', 'France', 'Finland', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Sweden', 'Slovakia', 'Slovenia', 'United Kingdom']
    countrycode_list = ['AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FR', 'FI', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SK', 'SI', 'UK']

    n_countries = len(country_list)
    index = 0
    id_country = False

    while index < n_countries:
        shape_records = sf.shapeRecords()
        desired_shapes = []
        for s in shape_records:
            if s.record[1] == country_list[index]:
                desired_shapes.append(s.shape)
            polygon = desired_shapes
            shpfilePoints = [shape.points for shape in polygon]
            polygons = shpfilePoints
        for polygon in polygons:
            poly = Polygon(polygon)
        p1 = Point(long, lat)
    
        country_code = countrycode_list[index]
        
        if p1.within(poly) == True:
            id_country = True
            break
        else:
            index += 1
        
    if id_country == True:
        return country_code
    else:
        print("Location not in compatible location")
        
#TODO uncomment for prod
#session_dict["country_code"] = countrycode()    

#------------------------------------------------------------------------------
# 1c. Get archetype input data for country
#------------------------------------------------------------------------------     
def input_select():
    country_code = session_dict["country_code"]
    design_choice = session_dict["design_choice"]
    conditioning = session_dict["conditioning"]

    #roof_bycountrycode = {'AT':0.22,'BE':0.57,'BG':0.3,'CY':0.55,'CZ':0.24,'DK':0.138655,'EE':0.175,'FI':0.18,'FR':0.21551,'DE':0.22,'GR':0.390526,'HU':0.25,'IE':0.310403,'IT':1.2,'LV':0.5,'LT':0.18,'LU':0.25,'MT':1.81,'NL':0.4,'PL':0.6,'PT':1.325,'RO':1,'SK':0.3,'SI':0.2,'ES':0.54,'SE':0.15,'UK':0.420262,'RS':0.45,'HR':0.29}
    #wall_bycountrycode = {'AT':0.45,'BE':0.78,'BG':0.425,'CY':1.39,'CZ':0.38,'DK':0.247311,'EE':0.225,'FI':0.26,'FR':0.380679,'DE':0.35,'GR':0.546737,'HU':0.45,'IE':0.324363,'IT':0.85,'LV':0.5,'LT':0.18,'LU':0.35,'MT':1.57,'NL':0.4,'PL':0.425,'PT':0.77,'RO':0.9,'SK':0.46,'SI':0.28,'ES':0.8,'SE':0.18,'UK':0.43689,'RS':0.9,'HR':0.34}
    #window_bycountrycode = {'AT':1.5,'BE':2.47,'BG':2.225,'CY':2.7,'CZ':1.7,'DK':1.710588,'EE':1.05,'FI':1.5,'FR':1.803397,'DE':1.6,'GR':3.124209,'HU':1.5,'IE':2.532219,'IT':4,'LV':1.8,'LT':1.62,'LU':1.3,'MT':5.8,'NL':2.012892,'PL':2.3,'PT':3.15,'RO':1.4,'SK':1.7,'SI':1.3,'ES':3.1,'SE':1.87,'UK':2.098009,'RS':2.6,'HR':1.8}
    
    roof_bycountrycode = {'AT':0.12,
                          'BE':0.12,
                          'BG':0.12,
                          'CY':0.12,
                          'CZ':0.12,
                          'DK':0.12,
                          'EE':0.12,
                          'FI':0.12,
                          'FR':0.12,
                          'DE':0.12,
                          'GR':0.12,
                          'HU':0.12,
                          'IE':0.12,
                          'IT':0.12,
                          'LV':0.12,
                          'LT':0.12,
                          'LU':0.12,
                          'MT':0.12,
                          'NL':0.12,
                          'PL':0.12,
                          'PT':0.12,
                          'RO':0.12,
                          'SK':0.12,
                          'SI':0.12,
                          'ES':0.12,
                          'SE':0.12,
                          'UK':0.12,
                          'RS':0.12,
                          'HR':0.12}
    wall_bycountrycode = {'AT':0.11,
                          'BE':0.11,
                          'BG':0.11,
                          'CY':0.11,
                          'CZ':0.11,
                          'DK':0.11,
                          'EE':0.11,
                          'FI':0.11,
                          'FR':0.11,
                          'DE':0.11,
                          'GR':0.11,
                          'HU':0.11,
                          'IE':0.11,
                          'IT':0.11,
                          'LV':0.11,
                          'LT':0.11,
                          'LU':0.11,
                          'MT':0.11,
                          'NL':0.11,
                          'PL':0.11,
                          'PT':0.11,
                          'RO':0.11,
                          'SK':0.11,
                          'SI':0.11,
                          'ES':0.11,
                          'SE':0.11,
                          'UK':0.11,
                          'RS':0.11,
                          'HR':0.11}
    window_bycountrycode = {'AT':1.3,
                          'BE':1.3,
                          'BG':1.3,
                          'CY':1.3,
                          'CZ':1.3,
                          'DK':1.3,
                          'EE':1.3,
                          'FI':1.3,
                          'FR':1.3,
                          'DE':1.3,
                          'GR':1.3,
                          'HU':1.3,
                          'IE':1.3,
                          'IT':1.3,
                          'LV':1.3,
                          'LT':1.3,
                          'LU':1.3,
                          'MT':1.3,
                          'NL':1.3,
                          'PL':1.3,
                          'PT':1.3,
                          'RO':1.3,
                          'SK':1.3,
                          'SI':1.3,
                          'ES':1.3,
                          'SE':1.3,
                          'UK':1.3,
                          'RS':1.3,
                          'HR':1.3}
    
    gfa_bycountrycode = {'AT':130.4232,'BE':73.238458,'BG':57.950391,'CY':130,'CZ':143.623088,'DK':165.15118,'EE':72.801848,'FI':120.244709,'FR':111.624939,'DE':122.27566,'GR':87.099868,'HU':93.149106,'IE':140.813219,'IT':111.157429,'LV':96,'LT':190.177304,'LU':99.298794,'MT':98.85399,'NL':116.733993,'PL':109.692701,'PT':151.594301,'RO':72.584769,'SK':107.235849,'SI':108.502771,'ES':158.116278,'SE':124.7,'UK':81.920004,'RS':99.445614,'HR':76}
    FEC_SH_bycountrycode = {'AT':104.242413,'BE':102.2565616,'BG':57.05747241,'HR':48.95924567,'CY':57.48413698,'CZ':109.6689323,'DK':50.91749641,'EE':168.4526184,'FI':113.0712994,'FR':72.42394807,'DE':64.51876319,'GR':87.1551152,'HU':69.84018057,'IE':73.40336728,'IT':86.02962418,'LV':120.9810201,'LT':61.35468261,'LU':41.00264632,'MT':16.37135061,'NL':13.34692239,'PL':97.78014375,'PT':82.27949335,'RO':73.85821177,'SK':108.1608934,'SI':75.35062828,'ES':45.92762111,'SE':85.84418576,'UK':53.56288364}
    FEC_SC_bycountrycode = {'AT':3.974299449,'BE':1.625205454,'BG':4.803880354,'HR':14.81262796,'CY':11.1598781,'CZ':3.262890936,'DK':7.345060615,'EE':7.528722395,'FI':7.759898605,'FR':9.304966333,'DE':5.689853376,'GR':30.12271817,'HU':6.558151171,'IE':2.676779788,'IT':38.02544447,'LV':12.69486181,'LT':1.361183611,'LU':5.232761639,'MT':20.40535646,'NL':4.610877322,'PL':3.716354612,'PT':12.78748966,'RO':3.728414016,'SK':4.518386324,'SI':8.842664135,'ES':11.85704718,'SE':8.542064266,'UK':3.502529005}
    #FEC_DHW_bycountrycode = {'AT':29.11329975,'BE':31.92694163,'BG':8.000344,'HR':31.449696,'CY':18.8998,'CZ':26.03714042,'DK':24.06287937,'EE':61.74479267,'FI':59.15112629,'FR':19.91263004,'DE':25.20299713,'GR':31.50579519,'HU':42.57736955,'IE':29.08293759,'IT':25.38933732,'LV':59.33706606,'LT':21.15401038,'LU':26.52911728,'MT':11.91406631,'NL':27.22221276,'PL':38.16994208,'PT':26.50432575,'RO':37.58787816,'SK':36.99244749,'SI':39.88805049,'ES':28.25489556,'SE':22.66189799,'UK':38.04675302}
    scenario_balanced_bycountrycode = {'AT':0.00714,'BE':0.008316,'BG':0.00525,'HR':0.007686,'CY':0.00357,'CZ':0.008442,'DK':0.007182,'EE':0.012348,'FI':0.009408,'FR':0.006762,'DE':0.007308,'GR':0.006048,'HU':0.007602,'IE':0.009156,'IT':0.00777,'LV':0.010962,'LT':0.005838,'LU':0.005964,'MT':0.003276,'NL':0.006132,'PL':0.008526,'PT':0.0063,'RO':0.006342,'SK':0.008694,'SI':0.008736,'ES':0.005838,'SE':0.007308,'UK':0.008736}
    scenario_low_co2_bycountrycode = {'AT':0.014322,'BE':0.014994,'BG':0.009954,'HR':0.014364,'CY':0.007518,'CZ':0.015246,'DK':0.01344,'EE':0.020034,'FI':0.016674,'FR':0.012474,'DE':0.012936,'GR':0.012012,'HU':0.013272,'IE':0.015498,'IT':0.01533,'LV':0.018228,'LT':0.011382,'LU':0.011172,'MT':0.006048,'NL':0.010752,'PL':0.015498,'PT':0.01092,'RO':0.01239,'SK':0.014448,'SI':0.013482,'ES':0.010206,'SE':0.014994,'UK':0.014196}
    scenario_self_sufficiency_bycountrycode = {'AT':0.023646,'BE':0.02604,'BG':0.021336,'HR':0.024528,'CY':0.020832,'CZ':0.027426,'DK':0.025704,'EE':0.033222,'FI':0.02877,'FR':0.02352,'DE':0.023898,'GR':0.022386,'HU':0.021966,'IE':0.026628,'IT':0.027972,'LV':0.02961,'LT':0.025032,'LU':0.020916,'MT':0.01596,'NL':0.026208,'PL':0.023436,'PT':0.026082,'RO':0.022722,'SK':0.02415,'SI':0.02877,'ES':0.023478,'SE':0.025956,'UK':0.02877}
    country_name__bycountrycode = {'AT':'Austria', 'BE':'Belgium', 'BG':'Bulgaria', 'HR':'Croatia', 'CY':'Cyprus', 'CZ':'Czech Republic', 'DE':'Germany', 'DK':'Denmark', 'EE':'Estonia', 'ES':'Spain', 'FR':'France', 'FI':'Finland', 'GR':'Greece', 'HU':'Hungary', 'IE':'Ireland', 'IT':'Italy', 'LV':'Latvia','LT':'Lithuania', 'LU':'Luxembourg', 'MT':'Malta', 'NL':'Netherlands', 'NO':'Norway', 'PL':'Poland', 'PT':'Portugal', 'RO':'Romania', 'SE':'Sweden', 'SK':'Slovakia', 'SI':'Slovenia', 'UK':'United Kingdom'}
    
    roof_uvalue = roof_bycountrycode[country_code]
    wall_uvalue = wall_bycountrycode[country_code]
    window_uvalue = window_bycountrycode[country_code]
    gfa = gfa_bycountrycode[country_code]
    gfa = gfa_bycountrycode[country_code]
    country_name = country_name__bycountrycode[country_code]
    
    FEC_SH = FEC_SH_bycountrycode[country_code]
    FEC_SC = FEC_SC_bycountrycode[country_code]
    #FEC_DHW = FEC_DHW_bycountrycode[country_code]
    scenario_balanced = scenario_balanced_bycountrycode[country_code]
    scenario_low_co2 = scenario_low_co2_bycountrycode[country_code]
    scenario_self_sufficiency = scenario_self_sufficiency_bycountrycode[country_code]
    
    if design_choice == "Self-sufficiency":
        normalized_nominal_power = scenario_self_sufficiency
    elif design_choice == "Balanced":
        normalized_nominal_power = scenario_balanced
    elif design_choice == "Low CO2":
        normalized_nominal_power = scenario_low_co2
    else:
        print("Invalid PV design choice")
        
    if conditioning == "Cooling":
        fec = FEC_SC
    elif conditioning == "Heating":
        fec = FEC_SH
    else:
        print("Invalid conditioning selection, must be Heating or Cooling")
    
    gfa_external = gfa / (pm["gfa_parameter"] * session_dict["n_stories"])
    tilt_radians = session_dict["roof_tilt"]*(np.pi/180)
    a_roof_total = gfa_external / np.cos(tilt_radians)
    
    nominal_power = round(normalized_nominal_power*a_roof_total,1)
    
    session_dict["u_wall_1"] = wall_uvalue
    session_dict["u_wall_north"] = wall_uvalue
    session_dict["u_wall_east"] = wall_uvalue
    session_dict["u_wall_south"] = wall_uvalue
    session_dict["u_wall_west"] = wall_uvalue
    session_dict["u_roof_1"] = roof_uvalue
    session_dict["u_roof_2"] = roof_uvalue
    session_dict["u_door_1"] = window_uvalue
    session_dict["u_window_1"] = window_uvalue
    session_dict["gfa"] = gfa
    session_dict["nominal_power"] = nominal_power
    session_dict["fec"] = fec 
    session_dict["country_name"] = country_name
    
    return roof_uvalue, wall_uvalue, window_uvalue, gfa, nominal_power, fec, country_name

#session_dict["roof_uvalue"],session_dict["wall_uvalue"],session_dict["window_uvalue"],session_dict["gfa"] = input_select()
"""
roof_uvalue, wall_uvalue, window_uvalue, gfa, nominal_power, fec, country_name = input_select()
session_dict["u_wall_1"] = wall_uvalue
session_dict["u_wall_north"] = wall_uvalue
session_dict["u_wall_east"] = wall_uvalue
session_dict["u_wall_south"] = wall_uvalue
session_dict["u_wall_west"] = wall_uvalue
session_dict["u_roof_1"] = roof_uvalue
session_dict["u_roof_2"] = roof_uvalue
session_dict["u_door_1"] = window_uvalue
session_dict["u_window_1"] = window_uvalue
session_dict["gfa"] = gfa
session_dict["nominal_power"] = nominal_power
session_dict["fec"] = fec 
session_dict["country_name"] = country_name
"""

#------------------------------------------------------------------------------
# 2a. Get solar gains input data 
#------------------------------------------------------------------------------
def get_pv():
    
    lat = session_dict["lat"]
    long = session_dict["long"]
    year = session_dict["year"]
    
    tilt = session_dict["roof_tilt"]
    azimuth = session_dict["roof_azimuth"]
    peak_power = session_dict["nominal_power"]
    
    start = session_dict["start"]
    end = session_dict["end"]
    
    losses = 14
    
    url = 'https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=' + str(lat) + '&lon=' + str(
        long) + '&startyear=' + str(year)+ '&endyear=' + str(year) + '&pvcalculation=' + str(1) + '&peakpower=' + str(peak_power) + '&loss=' + str(losses) + '&angle=' + str(
        tilt) + '&aspect=' + str(azimuth) + '&outputformat=csv&browser=1'
        
    r = requests.get(url, allow_redirects=True)
    pv_filename = r.headers['Content-disposition'].split('filename=')[1]
    pv_filename = 'PV data.csv'
    with open(pv_filename, 'wb') as pv_file:
        pv_file.write(r.content)
    
    pv_header_no_of_rows = 10
    pv_footer_no_of_rows = 10
    pv_weather_data = pd.read_csv(pv_filename, skiprows=pv_header_no_of_rows, skipfooter=pv_footer_no_of_rows,
                                    index_col=0, parse_dates=True,
                                    date_parser=lambda x: dt.datetime.strptime(x, "%Y%m%d:%H%M"), engine='python')
    
    pv_weather_data = pv_weather_data.tz_localize(None)
    pv_weather_data = pv_weather_data[start:end]
    
    pv_weather_data.index = pv_weather_data.index.map(lambda t: t.replace(minute=00))
    
    #pv_weather_data.to_csv('Output/pv_weather_data.csv')  #delete
    os.remove(pv_filename)
    return pv_weather_data

def get_pv_tmy():
    
    lat = session_dict["lat"]
    long = session_dict["long"]
    year = session_dict["year"]
    
    url = 'https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=' + str(lat) + '&lon=' + str(long) + '&outputformat=csv&browser=1'
            
    r = requests.get(url, allow_redirects=True)
    pv_filename = r.headers['Content-disposition'].split('filename=')[1]
    pv_filename = 'PV data tmy.csv'
    with open(pv_filename, 'wb') as pv_file:
        pv_file.write(r.content)
    
    pv_header_no_of_rows = 16
    pv_footer_no_of_rows = 12
    pv_weather_tmy_data = pd.read_csv(pv_filename, skiprows=pv_header_no_of_rows, skipfooter=pv_footer_no_of_rows,
                                    index_col=0, parse_dates=True,
                                    date_parser=lambda x: dt.datetime.strptime(x, "%Y%m%d:%H%M"), engine='python')
    
    pv_weather_tmy_data = pv_weather_tmy_data.tz_localize(None)
    pv_weather_tmy_data = pv_weather_tmy_data[pv_weather_tmy_data.index.month == session_dict["month_number"]] #DATE-X
    year_tmy = pv_weather_tmy_data.index.year[0]
    start = pd.Timestamp(str(year_tmy)+'-08-'+str(session_dict["start_day"])+' 00:00:00') #DATE-X
    end = pd.Timestamp(str(year_tmy)+'-08-'+str(session_dict["end_day"]-1)+' 23:00:00') #DATE-X
    pv_weather_tmy_data = pv_weather_tmy_data[start:end]    
    pv_weather_tmy_data.index = pv_weather_tmy_data.index.map(lambda t: t.replace(year=year))
    
    os.remove(pv_filename)
    return pv_weather_tmy_data

#------------------------------------------------------------------------------
# 2b. Calculate solar gains
#------------------------------------------------------------------------------
def solar_gains():

    #df_weather = df_weather_loc.join(humidity_join)
    lat = session_dict["lat"]
    long = session_dict["long"]
    year = session_dict["year"]
    
    gfa_interior = session_dict["gfa"]

    # User defined:
    # Number of stories
    n_stories = session_dict["n_stories"]  
    # Exterior gross floor area
    gfa_external = gfa_interior / (pm["gfa_parameter"] * n_stories)
    
    # Configure external elements
    
    # Set roof type and orientation
    # 0: gable roof; north/south orientation
    # 1: gable roof; east/west orientation
    # 2: skillion roof; north orientation
    # 3: skillion roof; east orientation
    # 4: skillion roof; south orientation
    # 5: skillion roof; west orientation
    # 6: flat roof; horizontal orientation
    roof_type_orientation = session_dict["roof_type_orientation"]
    # Orientation values
    orientation_lib = ["north","east","south","west"]
    
    # Set wall to floor ratio
    w_f_r = 1.30
    # Calculate external wall area
    a_wall_ext = w_f_r * gfa_external * n_stories
    # Set building shape (L:W) NOTE: L is length of building (sides), W is width of building (front/back)
    L = 1
    W = 1
    # Calculate building external wall areas based on shape (NOTE: wall side_1 is left wall when facing facade)
    session_dict["a_external_front"] = a_external_front = ((W/(L+W)) * a_wall_ext)/2
    session_dict["a_external_back"] = a_external_back = ((W/(L+W)) * a_wall_ext)/2
    session_dict["a_external_side_1"] = a_external_side_1 = ((L/(L+W)) * a_wall_ext)/2
    session_dict["a_external_side_2"] = a_external_side_2 = ((L/(L+W)) * a_wall_ext)/2
    
    # Set building shape and facade orientation
    # Select facade orientation (0: north, 1: east, 2: south, 3: west)
    facade_orientation = orientation_lib[0]
    
    # Set area of windows (as % of respective wall or roof)
    window_front_proportion = 10
    window_back_proportion = 25
    window_side_1_proportion = 25
    window_side_2_proportion = 25
    window_roof_proportion = 0
    
    # Set area of door (m2)
    a_door_1 = 2
    session_dict["a_door_1"] = a_door_1
    
    # Set roof pitch (angle in degrees)
    user_roof_pitch = session_dict["roof_tilt"]
    
    # Ensure flat roof has 0 degree pitch
    if roof_type_orientation == 6:
        roof_pitch = 0
    else:
        roof_pitch = user_roof_pitch
    # Calculate roof area based on external structure area and pitch
    a_roof_total = gfa_external / np.cos(roof_pitch)
    
    # Calculate area of window
    a_window_front = (window_front_proportion/100) * (a_external_front - a_door_1)
    a_window_back = (window_back_proportion/100) * a_external_back
    a_window_side_1 = (window_side_1_proportion/100) * a_external_side_1
    a_window_side_2 = (window_side_2_proportion/100) * a_external_side_2
    
    # Calculate area of external walls
    a_wall_front = a_external_front - a_window_front - a_door_1
    a_wall_back = a_external_back - a_window_back
    a_wall_side_1 = a_external_side_1 - a_window_side_1
    a_wall_side_2 = a_external_side_2 - a_window_side_2
    
    if facade_orientation == "north":
        session_dict["door_1_azimuth"] = 0.0
        session_dict["a_window_north"] = a_window_front
        session_dict["a_window_south"] = a_window_back
        session_dict["a_window_west"] = a_window_side_1
        session_dict["a_window_east"] = a_window_side_2
        session_dict["a_wall_north"] = a_wall_front
        session_dict["a_wall_south"] = a_wall_back
        session_dict["a_wall_west"] = a_wall_side_1
        session_dict["a_wall_east"] = a_wall_side_2
    elif facade_orientation == "south":
        session_dict["door_1_azimuth"] = 180.0
        session_dict["a_window_south"] = a_window_front
        session_dict["a_window_north"] = a_window_back
        session_dict["a_window_east"] = a_window_side_1
        session_dict["a_window_west"] = a_window_side_2
        session_dict["a_wall_south"] = a_wall_front
        session_dict["a_wall_north"] = a_wall_back
        session_dict["a_wall_east"] = a_wall_side_1
        session_dict["a_wall_west"] = a_wall_side_2
    elif facade_orientation == "east":
        session_dict["door_1_azimuth"] = 90.0
        session_dict["a_window_east"] = a_window_front
        session_dict["a_window_west"] = a_window_back
        session_dict["a_window_north"] = a_window_side_1
        session_dict["a_window_south"] = a_window_side_2
        session_dict["a_wall_east"] = a_wall_front
        session_dict["a_wall_west"] = a_wall_back
        session_dict["a_wall_north"] = a_wall_side_1
        session_dict["a_wall_south"] = a_wall_side_2    
    else:
        session_dict["door_1_azimuth"] = 270.0
        session_dict["a_window_west"] = a_window_front
        session_dict["a_window_east"] = a_window_back
        session_dict["a_window_south"] = a_window_side_1
        session_dict["a_window_north"] = a_window_side_2
        session_dict["a_wall_west"] = a_wall_front
        session_dict["a_wall_east"] = a_wall_back
        session_dict["a_wall_south"] = a_wall_side_1
        session_dict["a_wall_north"] = a_wall_side_2
    
    # Roof type, tilt, and orientation
    if roof_type_orientation == 0:
        session_dict["roof_1_azimuth"] = 0.0
        session_dict["roof_2_azimuth"] = 180.0
        session_dict["a_roof_1"] = a_roof_total/2
        session_dict["a_roof_2"] = a_roof_total/2
    elif roof_type_orientation == 1:
        session_dict["roof_1_azimuth"] = 90.0
        session_dict["roof_2_azimuth"] = 270.0    
        session_dict["a_roof_1"] = a_roof_total/2
        session_dict["a_roof_2"] = a_roof_total/2
    elif roof_type_orientation == 2:
        session_dict["roof_1_azimuth"] = 0.0
        session_dict["roof_2_azimuth"] = 0.0
        session_dict["a_roof_1"] = a_roof_total
        session_dict["a_roof_2"] = 0.0
    elif roof_type_orientation == 3:
        session_dict["roof_1_azimuth"] = 90.0
        session_dict["roof_2_azimuth"] = 90.0
        session_dict["a_roof_1"] = a_roof_total
        session_dict["a_roof_2"] = 0.0
    elif roof_type_orientation == 4:
        session_dict["roof_1_azimuth"] = 180.0
        session_dict["roof_2_azimuth"] = 180.
        session_dict["a_roof_1"] = a_roof_total
        session_dict["a_roof_2"] = 0.0
    elif roof_type_orientation == 5:
        session_dict["roof_1_azimuth"] = 270.0
        session_dict["roof_2_azimuth"] = 270.0
        session_dict["a_roof_1"] = a_roof_total
        session_dict["a_roof_2"] = 0.0
    elif roof_type_orientation == 6:
        session_dict["roof_1_azimuth"] = 0.0
        session_dict["roof_2_azimuth"] = 0.0
        session_dict["a_roof_1"] = a_roof_total
        session_dict["a_roof_2"] = 0.0
    else:
        print("Invalid Roof Type/Orientation code")
    
    
    element = ["window_north","window_east","window_south","window_west","door_1","roof_1","roof_2","wall_north","wall_east","wall_south","wall_west"]
    azimuth_lib = {"north": 0.0,"east": 90.0,"south": 180.0,"west": 270.0,"door_1": session_dict["door_1_azimuth"],"roof_1": session_dict["roof_1_azimuth"],"roof_2": session_dict["roof_2_azimuth"]}
    tilt_lib = {"north": 90.0,"east": 90.0,"south": 90.0,"west": 90.0,"door_1": 90.0,"roof_1": roof_pitch,"roof_2": roof_pitch}   

    # Conditioned floor area (m2)
    session_dict["A_f"] = A_f = gfa_interior
    # Effective mass area (m2)
    session_dict["A_m"] = A_m = pm["A_m_parameter"] * A_f
    # Thermal capacitance of medium (J/K)
    session_dict["C_m"] = C_m = pm["C_m_parameter"] * A_f
    # Area of all surfaces facing the building zone (m2)
    A_tot = pm["lambda_at"] * A_f
    session_dict["A_t"] = A_t = A_tot
    # Coupling conductance between air and surface nodes (W/K)
    session_dict["H_tr_is"] = H_tr_is = pm["h_is"] * A_tot
    
    # Ventilation heat transfer coefficients
    # Total air changes (1/h)
    session_dict["n_air_total"] = n_air_total = pm["n_air_inf"] + pm["n_air_use"]
    # Air volume of room (m3)
    session_dict["air_volume"] = air_volume = pm["h_room"] * A_f
    # Temperature adjustment factor for the air flow element
    session_dict["b_ve"] = b_ve = 1
    # Time fraction of operation of the air flow element (f_ve_t = 1 assumes element is always operating)
    # E.g. If only operating from 8.00 to 18.00, then f_ve_t = 10 hours/24 hours = 0.417
    session_dict["f_ve_t"] = f_ve_t = 1
    # Airflow rate of the air flow element (m3/s)
    session_dict["q_ve"] = q_ve = n_air_total/3600 * air_volume
    # Time-average airflow rate of the air flow element (m3/s)
    session_dict["q_ve_mn"] = q_ve_mn = f_ve_t * q_ve
    # Ventilation heat transfer coefficient (W/K)
    session_dict["H_ve_adj"] = H_ve_adj = pm["rho_a_c_a"] * b_ve * q_ve_mn
    # Combined heat conductance (W/K)
    session_dict["H_tr_1"] = H_tr_1 = 1 / (1/H_ve_adj + 1/H_tr_is)
    
    # Heat transfer opaque elements (walls and roof) (W/K)
    session_dict["H_tr_op"] = H_tr_op = (a_wall_front + a_wall_back + a_wall_side_1 + a_wall_side_2) * session_dict["u_wall_1"] + a_roof_total * session_dict["u_roof_1"]
    # Thermal transmission coefficient of glazed elements (windows and doors) (W/K)
    session_dict["H_tr_w"] = H_tr_w = a_window_front * session_dict["u_window_1"] + a_window_back * session_dict["u_window_1"] + a_window_side_1 * session_dict["u_window_1"] + a_window_side_2 * session_dict["u_window_1"] + a_door_1 * session_dict["u_door_1"]
    # Split H_tr_op into coupling conductance between nodes m and s (W/K)
    session_dict["H_op"] = H_op = H_tr_op
    session_dict["H_ms"] = H_ms = pm["h_ms"] * A_m
    session_dict["H_em"] = H_em = 1/(1/H_op - 1/H_ms)
    session_dict["H_tr_ms"] = H_tr_ms = H_ms
    session_dict["H_tr_em"] = H_tr_em = H_em
    
    # External radiative heat transfer coefficient for opaque and glazed surfaces (W/(m²K))
    h_r_op = 4 * pm["epsilon_op"] * pm["SB_constant"] * ((pm["theta_ss"] + 273)**3)
    h_r_gl = 4 * pm["epsilon_gl"] * pm["SB_constant"] * ((pm["theta_ss"] + 273)**3)
    
    # Set up date/time and list of inputs/outputs for calculaton
    start_date = dt.datetime(2020,8,session_dict["start_day"],0) #DATE-X
    end_date = dt.datetime(2020,8,session_dict["end_day"]-1,23) #DATE-X
    delta = dt.timedelta(hours=1)
    time_length = end_date - start_date
    cols_g = ['hour_of_day','day_of_week','t_outside_t2m','t_outside_d2m','humidity','surface_pressure','ghi','dni','dhi','solar_position','relative_airmass','extra_radiation','solar_altitude','solar_altitude_radians','solar_azimuth','solar_azimuth_radians','solar_zenith','solar_zenith_radians','total_irradiance','phi_int','phi_sol','Q_HC_nd','Q_H_nd','Q_C_nd','theta_air','t_outside_t2m_degC','theta_air_ac', 'i_sol', 'theta_air_0', 'theta_air_10', 'theta_m_tp', 'theta_m_t', 'phi_ia', 'phi_st', 'phi_m', 'pv_power']
    n_timesteps = int(time_length.total_seconds() / delta.total_seconds())
    timestamp_list = [(start_date + x*delta) for x in range(0,n_timesteps+1)]
    #timestamp_list = [(start_date + x*delta) for x in range(-1,n_timesteps+1)] #vvvv
    df_g = pd.DataFrame(columns=cols_g,index=timestamp_list)
    
    pv_data = get_pv()  # get PV data from PVGIS website
    pv_data_tmy = get_pv_tmy()  # get PV data from PVGIS website
    df_weather = pd.merge(pv_data, pv_data_tmy, left_index=True, right_index=True)
    df_weather.rename(columns={"G(i)": "Gi", "G(h)": "Gh", "Gb(n)": "Gbn", "Gd(h)": "Gdh", "IR(h)": "IRh"}, inplace = True)
    
    current_timestamp = start_date
    #current_timestamp = start_date - delta #vvvv
    current_index = 0
    #current_index = -1 #vvvv
    
    # Build DF with solar radiation data
    while current_timestamp <= end_date:
    
        df_g.t_outside_t2m[current_index] = df_weather.T2m_x[current_index]

        # Relative humidity
        df_g.humidity[current_index] = df_weather.RH[current_index]
        # Surface pressure
        df_g.surface_pressure[current_index] = df_weather.SP[current_index]
        # Global Horizontal Irradiance (GHI)
        df_g.ghi[current_index] = df_weather.Gh[current_index]
        ghi = df_g.ghi[current_index]
        # Direct Normal Irradiance (DNI)
        df_g.dni[current_index] = df_weather.Gbn[current_index]
        dni = df_g.dni[current_index]
        # Diffuse Horizontal Irradiance (DHI)
        df_g.dhi[current_index] = df_weather.Gdh[current_index]
        dhi = df_g.dhi[current_index]
        
        # Solar position
        df_g.solar_position[current_index] = pvlib.solarposition.get_solarposition(current_timestamp, lat, long, pressure=df_g.surface_pressure[current_index], temperature=df_g.t_outside_t2m[current_index])
        solar_position = df_g.solar_position[current_index]
        # Solar azimuth
        df_g.solar_azimuth[current_index] = solar_position["azimuth"]
        solar_azimuth = df_g.solar_azimuth[current_index]
        # Solar zenith
        df_g.solar_zenith[current_index] = solar_position["apparent_zenith"]
        solar_zenith = df_g.solar_zenith[current_index]
        # Airmass
        df_g.relative_airmass[current_index] = pvlib.atmosphere.get_relative_airmass(solar_zenith)
        relative_airmass = df_g.relative_airmass[current_index]
        # Extra DNI
        df_g.extra_radiation[current_index] = pvlib.irradiance.get_extra_radiation(current_timestamp.timetuple().tm_yday, epoch_year = current_timestamp. year)
        dni_extra = df_g.extra_radiation[current_index] 
        
        # PV power output
        df_g.pv_power[current_index] = df_weather.P[current_index]
        #df_g.pv_power[current_index] = abs(df_weather.P[current_index]*-1+5000) #TESTING
                
        
        # Calculates values for the form factor for radiation between the unshaded roof and the sky
        F_r_roof = 1 - roof_pitch/180
        # Solar radiation and gains results storage
        phi_sol_results = []
        i_sol_results = []
        
        for i in azimuth_lib:
            surface_azimuth = azimuth_lib[i]
            surface_tilt = tilt_lib[i]
            I_sol_element = pvlib.irradiance.get_total_irradiance(
                        surface_tilt, 
                        surface_azimuth,
                        solar_zenith,
                        solar_azimuth,
                        dni=dni,
                        ghi=ghi,
                        dhi=dhi,
                        dni_extra=dni_extra,
                        airmass=relative_airmass,                                                                                  
                        model="perez")
            # Irradiance on element (W/m2)
            session_dict[f"I_sol_{i}"] = float(I_sol_element["poa_global"].fillna(0))
            i_sol_results.append(float(session_dict[f"I_sol_{i}"]))
        
        session_dict["i_sol_window_north"] = i_sol_results[0]
        session_dict["i_sol_window_east"] = i_sol_results[1]
        session_dict["i_sol_window_south"] = i_sol_results[2]
        session_dict["i_sol_window_west"] = i_sol_results[3]
        session_dict["i_sol_vert"] = (i_sol_results[0] + i_sol_results[1] + i_sol_results[2] + i_sol_results[3])/4
        session_dict["i_sol_door_1"] = i_sol_results[4]
        session_dict["i_sol_roof_1"] = i_sol_results[5]
        session_dict["i_sol_roof_2"] = i_sol_results[6]
        session_dict["i_sol_wall_north"] = i_sol_results[0]
        session_dict["i_sol_wall_east"] = i_sol_results[1]
        session_dict["i_sol_wall_south"] = i_sol_results[2]
        session_dict["i_sol_wall_west"] = i_sol_results[3]
                
        for i in element:
            element_name_a = "a_" + i
            element_area = session_dict[f"a_{i}"]
            g_gl = pm["F_w"] * pm["g_gl_n"]
            if (i == "window_north") | (i == "window_east") | (i == "window_south") | (i == "window_west"):
                element_name_i_sol = "i_sol_" + i
                i_sol_window = element_name_i_sol
                # Shading reduction factor for movable shading provisions
                F_sh_gl = pm["F_sh_vert"]
                element_u = session_dict["u_window_1"]  
                element_r = pm["R_S_e_gl"]
                # Effective solar collecting area of element
                session_dict[f"A_sol_{i}"] = F_sh_gl * g_gl * (1 - pm["F_F"]) * element_area
                # Thermal radiation heat flow to the sky (W)
                session_dict[f"phi_r_{i}"] = element_r * element_u * element_area * h_r_gl * pm["delta_theta_er"]
                # Heat flow by solar gains through building element
                session_dict[f"phi_sol_{i}"] = pm["F_sh_vert"] * session_dict[f"A_sol_{i}"] * session_dict[f"i_sol_{i}"] - pm["F_r_vert"] * session_dict[f"phi_r_{i}"]
                if float(session_dict[f"phi_sol_{i}"]) < 0:
                    session_dict[f"phi_sol_{i}"] = 0
                phi_sol_results.append(float(session_dict[f"phi_sol_{i}"]))
            elif (i == "door_1"):
                # Shading reduction factor for movable shading provisions
                F_sh_gl = pm["F_sh_vert"]
                element_name_u = "u_" + i
                element_u = session_dict[element_name_u]
                element_r = pm["R_S_e_gl"]
                # Effective solar collecting area of element
                session_dict[f"A_sol_{i}"] = F_sh_gl * g_gl * (1 - pm["F_F"]) * element_area
                # Thermal radiation heat flow to the sky (W)
                session_dict[f"phi_r_{i}"] = element_r * element_u * element_area * h_r_gl * pm["delta_theta_er"]
                # Heat flow by solar gains through building element
                session_dict[f"phi_sol_{i}"] = pm["F_sh_vert"] * session_dict[f"A_sol_{i}"] * session_dict["i_sol_door_1"] - pm["F_r_vert"] * session_dict[f"phi_r_{i}"]
                if float(session_dict[f"phi_sol_{i}"]) < 0:
                    session_dict[f"phi_sol_{i}"] = 0
                phi_sol_results.append(float(session_dict[f"phi_sol_{i}"]))
            elif (i == "roof_1") | (i == "roof_2"):
                element_name_i_sol = "i_sol_" + i
                i_sol_roof = element_name_i_sol
                element_name_u = "u_" + i
                element_u = session_dict[element_name_u]
                element_r = pm["R_S_e_op"]
                # Effective solar collecting area of element
                session_dict[f"A_sol_{i}"] = pm["alpha_S_c"] * element_r * element_u * element_area
                # Thermal radiation heat flow to the sky (W)
                session_dict[f"phi_r_{i}"] = element_r * element_u * element_area * h_r_op * pm["delta_theta_er"]
                # Heat flow by solar gains through building element
                session_dict[f"phi_sol_{i}"] = pm["F_sh_hor"] * session_dict[f"A_sol_{i}"] * session_dict[f"I_sol_{i}"] - F_r_roof * session_dict[f"phi_r_{i}"]
                if float(session_dict[f"phi_sol_{i}"]) < 0:
                    session_dict[f"phi_sol_{i}"] = 0
                phi_sol_results.append(float(session_dict[f"phi_sol_{i}"]))
            elif (i == "wall_north") | (i == "wall_east") | (i == "wall_south") | (i == "wall_west"):
                element_name_u = "u_" + i
                element_u = session_dict[element_name_u]
                element_r = pm["R_S_e_op"]
                # Effective solar collecting area of element 
                session_dict[f"A_sol_{i}"] = pm["alpha_S_c"] * element_r * element_u * element_area
                # Thermal radiation heat flow to the sky (W)
                session_dict[f"phi_r_{i}"] = element_r * element_u * element_area * h_r_op * pm["delta_theta_er"]
                # Heat flow by solar gains through building element
                session_dict[f"phi_sol_{i}"] = pm["F_sh_vert"] * session_dict[f"A_sol_{i}"] * session_dict[f"i_sol_{i}"] - pm["F_r_vert"] * session_dict[f"phi_r_{i}"]
                if float(session_dict[f"phi_sol_{i}"]) < 0:
                    session_dict[f"phi_sol_{i}"] = 0
                phi_sol_results.append(float(session_dict[f"phi_sol_{i}"]))
        
        df_g.i_sol[current_index] = sum(i_sol_results)
        df_g.phi_sol[current_index] = sum(phi_sol_results)   
        phi_sol = df_g.phi_sol[current_index]
        
        #INTERNAL GAINS
        # From Table G.8
        # Check if weekday or weekend (Monday-Friday = 0-4, Saturday-Sunday = 5-6)
        # Adjust heat flow rate from occupants and appliances in living room and kitchen (hf_Oc_A_LRK)
        # Adjust heat flow rate from occupants and appliances in other rooms (e.g. bedrooms) (hf_Oc_A_Oth)
        df_g.day_of_week[current_index] = df_g.index[current_index].weekday()
        df_g.hour_of_day[current_index] = df_g.index[current_index].hour
        day_of_week = df_g.index[current_index].weekday()
        hour_of_day = df_g.index[current_index].hour    
        if ((day_of_week == 0) | (day_of_week == 1) | (day_of_week == 2) | (day_of_week == 3) | (day_of_week == 4)) & (7 <= hour_of_day <= 17):
            hf_Oc_A_LRK = 8
            hf_Oc_A_Oth = 1
        elif ((day_of_week == 0) | (day_of_week == 1) | (day_of_week == 2) | (day_of_week == 3) | (day_of_week == 4)) & (17 < hour_of_day <= 23):
            hf_Oc_A_LRK = 20
            hf_Oc_A_Oth = 1
        elif ((day_of_week == 0) | (day_of_week == 1) | (day_of_week == 2) | (day_of_week == 3) | (day_of_week == 4)) & (23 < hour_of_day < 7):
            hf_Oc_A_LRK = 2
            hf_Oc_A_Oth = 6
        elif ((day_of_week == 5) | (day_of_week == 6)) & (7 <= hour_of_day <= 17):
            hf_Oc_A_LRK = 8
            hf_Oc_A_Oth = 2
        elif ((day_of_week == 5) | (day_of_week == 6)) & (17 < hour_of_day <= 23):
            hf_Oc_A_LRK = 20
            hf_Oc_A_Oth = 4
        else:
            hf_Oc_A_LRK = 2
            hf_Oc_A_Oth = 6    
     
        # Heat flow rate for metabolic heat from occupants and dissipated heat from appliances (W)
        phi_int_Oc_A = (hf_Oc_A_LRK + hf_Oc_A_Oth) * A_f
        # Heat flow rate from internal sources (W)
        df_g.phi_int[current_index] = float(phi_int_Oc_A)
        phi_int = df_g.phi_int[current_index]
        # Heat flow rate to air (W)
        df_g.phi_ia[current_index] = 0.5 * phi_int
        # Heat flow rate to internal surface (W)
        df_g.phi_st[current_index] = (1 - (A_m / A_t) - (H_tr_w / (9.1 * A_t))) * (0.5 * phi_int + phi_sol)
        df_g.phi_st[current_index] = float(df_g.phi_st[current_index])
        # Heat flow rate to medium (W)
        df_g.phi_m[current_index] = (A_m / A_t) * (0.5 * phi_int + phi_sol)    
        
        current_timestamp += delta
        current_index += 1  
    

    results = {'theta_out_degC':df_g.t_outside_t2m,'phi_ia':df_g.phi_ia,'phi_st': df_g.phi_st,'phi_m': df_g.phi_m,'pv_power':df_g.pv_power,
               'phi_int':df_g.phi_int, 'phi_sol':df_g.phi_sol
               }
    df_results = pd.DataFrame(results)
    
    return df_results


#------------------------------------------------------------------------------
# 3a. Prepare data for optimization
#------------------------------------------------------------------------------

def load_inputs():
    df_sg = solar_gains()

    df_sg.index = pd.to_datetime(df_sg.index, utc=True, errors='coerce') #zzzzz
    df_sg['theta_out_K'] = df_sg.theta_out_degC + 273.15
    df_sg = df_sg.tz_localize(None)    

    timestep_list_length = list(range(0, len(df_sg)))
    #timestep_list_length = list(range(-1, len(df_sg))) #vvvv
    dict_sg = df_sg.reset_index().to_dict()

    return timestep_list_length, df_sg, dict_sg

def define_length():
    param_dict.clear()
    param_dict["length"] = "week"
    end = 24
    eta_v = 0.87
    f_ve_t = 0.0833
    param_dict["eta_v"] = eta_v
    param_dict["f_ve_t"] = f_ve_t
    return end, eta_v, f_ve_t

#------------------------------------------------------------------------------
# 3b. Setup optimization model constraints and objective function
#------------------------------------------------------------------------------

# Objective functions

#------PV Match-------
def objective_function_pv_match(m):
    return sum(m.M_positive[t] + m.M_negative[t] for t in m.Time) #pv match

def objective_function_simple(m):
    return sum(-1*m.Q_hp_sh[t] for t in m.Time) #energy min

def match_constraint(m, t): 
    return m.match[t] <= m.pv_power[t]

def mismatch_constraint(m, t): 
    return -1*m.Q_hp_sh[t] - m.match[t] == m.M_positive[t] - m.M_negative[t]

#------Peak Shave-------

def objective_function_peak_shave(m):
    return m.z #zzzzz

def min_max_constraint(m, t): #zzzzz
    #return (m.z >= sum(m.Q_hp_sh[t] for t in m.Time))
    return m.z >= -1*m.Q_hp_sh[t]

#-------BAU (temperature match)------

def objective_function_bau(m):
    return sum(m.M_positive[t] + m.M_negative[t] for t in m.Time)

def match_constraint_bau(m, t): 
    return m.match[t] <= np.log(m.theta_air_outdoor_degC[t])

def mismatch_constraint_bau(m, t): 
    return -1*m.Q_hp_sh[t] - m.match[t] == m.M_positive[t] - m.M_negative[t]

#-------------

def rc_simulation_temperature_constraint(m, t):
    
    C_m = session_dict["C_m"]
    H_tr_is = session_dict["H_tr_is"]
    H_ve_adj = session_dict["H_ve_adj"]
    H_tr_1 = session_dict["H_tr_1"]
    H_tr_w = session_dict["H_tr_w"]
    H_tr_ms = session_dict["H_tr_ms"]
    H_tr_em = session_dict["H_tr_em"]
    
    eta_v = param_dict["eta_v"]

    # Set supply temperature equal to outside temperature
    theta_e = m.theta_air_outdoor_degC[t] #degC

    if t == 0:
        theta_m_tp = m.T_init  - 273.15 #degC
        
    else:
        theta_m_tp = m.theta_air_indoor[t-1] - 273.15 #degC
    
    theta_sup = eta_v * theta_m_tp + (1 - eta_v) * theta_e

    phi_ia = m.phi_ia[t] #W
    phi_st = m.phi_st[t] #W
    phi_m = m.phi_m[t] #W

    H_tr_2 = H_tr_1 + H_tr_w #W/K
    H_tr_3 = 1 / (1 / H_tr_2 + 1 / H_tr_ms) #W/K
    
    # Other combined heat conductances
    phi_HC_nd = m.Q_hp_sh[t-1] #W
    
    phi_mtot = phi_m + H_tr_em*theta_e + (H_tr_3*(phi_st + H_tr_w*theta_e + H_tr_1*(
            ((phi_ia + phi_HC_nd) / H_ve_adj) + theta_sup)) / H_tr_2) #W
    
    #building internal heat capacity
    int_heat_cap_bldg = (C_m / 3600) #J/K --> Wh/K
    heat_transfer = 0.5 * (H_tr_3 + H_tr_em) #W/K
    
    theta_m_t = (theta_m_tp * (int_heat_cap_bldg - heat_transfer) + phi_mtot) / (
            int_heat_cap_bldg + heat_transfer) #degC

    theta_m = (theta_m_t + theta_m_tp) / 2 #degC

    theta_s = (H_tr_ms * theta_m + phi_st + H_tr_w * theta_e + H_tr_1 * (
            theta_sup + (phi_ia + phi_HC_nd) / H_ve_adj)) / (H_tr_ms + H_tr_w + H_tr_1) #degC

    theta_air_ac = (H_tr_is * theta_s + H_ve_adj * theta_sup + phi_ia + phi_HC_nd) / (H_tr_is + H_ve_adj) #degC

    theta_air_ac = theta_air_ac + 273.15 #convert to K
        
    return m.theta_air_indoor[t] == theta_air_ac #K


#------------------------------------------------------------------------------
# 3c. Setup optimization model to calculate cooling demand
#------------------------------------------------------------------------------
def opt_model():    
    timestep_list_length, df_sg, dict_sg = load_inputs()
    
    end, eta_v, f_ve_t = define_length()
        
    m = en.ConcreteModel()

    # Sets
    m.Time = en.Set(initialize=timestep_list_length, ordered=True)
    m.tm = en.Set(initialize=[-1] + timestep_list_length, ordered=True)

    m.T_init = session_dict["ubound"] + 273.15 #293

    # Solar Gains
    m.phi_ia = en.Param(m.Time, initialize=dict_sg["phi_ia"])
    m.phi_st = en.Param(m.Time, initialize=dict_sg["phi_st"])
    m.phi_m = en.Param(m.Time, initialize=dict_sg["phi_m"])
    
    # PV paramaters
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])

    # Heating parameters 
    m.theta_air_outdoor = en.Param(m.Time, initialize=dict_sg["theta_out_K"])
    m.theta_air_outdoor_degC = en.Param(m.Time, initialize=dict_sg["theta_out_degC"])

    m.theta_air_indoor = en.Var(m.tm,
                                bounds=(session_dict["lbound"] + 273.15, session_dict["ubound"] + 273.15), #cooling test
                                initialize=m.T_init
                               )

    m.Q_hp_sh = en.Var(m.tm, bounds=(-20_000, 0), initialize=0) #W
    
    m.z = en.Var(domain=en.NonNegativeReals) #zzzzz
    m.min_max = en.Constraint(m.tm, rule=min_max_constraint) #zzzzz
    
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])
    m.match = en.Var(m.Time, domain=en.NonNegativeReals)
    m.M_positive = en.Var(m.Time, domain=en.NonNegativeReals)  #Excess demand
    m.M_negative = en.Var(m.Time, domain=en.NonNegativeReals)  #Excess supply
    m.supply_constraint = en.Constraint(m.Time, rule=match_constraint) 
    m.mismatch_constraint = en.Constraint(m.Time, rule=mismatch_constraint)
    m.rc_model_internal_temperature_r = en.Constraint(m.Time, rule=rc_simulation_temperature_constraint)
 
    # Objective Function and constraints
    if session_dict["scenario"] == "PV Match":
        m.match_pv_cooling = en.Objective(rule=objective_function_pv_match, sense=en.minimize)
    elif session_dict["scenario"] == "Peak Shaving":
        m.match_pv_cooling = en.Objective(rule=objective_function_peak_shave, sense=en.minimize)
    else:
        print("Invalid scenario")
    
    opt = SolverFactory('glpk') #cbc
    results = opt.solve(m, tee=True,symbolic_solver_labels=True)
    #m.load(results)    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        #print("*************Using regular solve*************")
        results = results
    else:
        print("Infeasible problem")
        pass
    
    results_table = pd.DataFrame({
        'Q_hp_sh': [-1*(m.Q_hp_sh[v].value) for v in m.Q_hp_sh][1:],
        'pv_power': [m.pv_power[i] for i in m.pv_power],
        'theta_air_indoor': [m.theta_air_indoor[v].value-273.15 for v in m.theta_air_indoor][1:],
        'theta_air_outdoor':  [m.theta_air_outdoor[v]-273.15 for v in m.theta_air_outdoor],
        'matched': [en.value(m.match[v]) for v in m.match]
    },
    index=df_sg.index)
    
    return results_table



def opt_model_simple():    
    timestep_list_length, df_sg, dict_sg = load_inputs()
    
    end, eta_v, f_ve_t = define_length()
        
    m = en.ConcreteModel()

    # Sets
    m.Time = en.Set(initialize=timestep_list_length, ordered=True)
    m.tm = en.Set(initialize=[-1] + timestep_list_length, ordered=True)

    m.T_init = session_dict["ubound"] + 273.15 #293

    # Solar Gains
    m.phi_ia = en.Param(m.Time, initialize=dict_sg["phi_ia"])
    m.phi_st = en.Param(m.Time, initialize=dict_sg["phi_st"])
    m.phi_m = en.Param(m.Time, initialize=dict_sg["phi_m"])
    
    # PV paramaters
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])

    # Heating parameters 
    m.theta_air_outdoor = en.Param(m.Time, initialize=dict_sg["theta_out_K"])
    m.theta_air_outdoor_degC = en.Param(m.Time, initialize=dict_sg["theta_out_degC"])

    m.theta_air_indoor = en.Var(m.tm,
                                bounds=(session_dict["lbound"] + 273.15, session_dict["ubound"] + 273.15), #cooling test
                                initialize=m.T_init
                               )

    m.Q_hp_sh = en.Var(m.tm, bounds=(-20_000, 0), initialize=0) #W
    
    m.z = en.Var(domain=en.NonNegativeReals) #zzzzz
    m.min_max = en.Constraint(m.tm, rule=min_max_constraint) #zzzzz
    
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])
    m.match = en.Var(m.Time, domain=en.NonNegativeReals)
    m.M_positive = en.Var(m.Time, domain=en.NonNegativeReals)  # Positive deviation (excess demand)
    m.M_negative = en.Var(m.Time, domain=en.NonNegativeReals)  # Negative deviation (excess supply)
    m.supply_constraint = en.Constraint(m.Time, rule=match_constraint)
    m.mismatch_constraint = en.Constraint(m.Time, rule=mismatch_constraint)
    m.rc_model_internal_temperature_r = en.Constraint(m.Time, rule=rc_simulation_temperature_constraint)
 
    # Objective Function and constraints
    m.match_pv_cooling = en.Objective(rule=objective_function_simple, sense=en.minimize)
    
    opt = SolverFactory('glpk') #cbc
    results = opt.solve(m, tee=True,symbolic_solver_labels=True)
    
    results_table = pd.DataFrame({
        'Q_C_nd_BAU_sim': [-1*(m.Q_hp_sh[v].value) for v in m.Q_hp_sh][1:],
        #'pv_power': [m.pv_power[i] for i in m.pv_power],
        #'theta_air_indoor': [m.theta_air_indoor[v].value-273.15 for v in m.theta_air_indoor][1:],
        #'theta_air_outdoor':  [m.theta_air_outdoor[v]-273.15 for v in m.theta_air_outdoor],
        #'matched': [en.value(m.match[v]) for v in m.match]
    },
    index=df_sg.index)
    
    return results_table

def opt_model_bau():    
    timestep_list_length, df_sg, dict_sg = load_inputs()
    
    end, eta_v, f_ve_t = define_length()
        
    m = en.ConcreteModel()

    # Sets
    m.Time = en.Set(initialize=timestep_list_length, ordered=True)
    m.tm = en.Set(initialize=[-1] + timestep_list_length, ordered=True)

    m.T_init = session_dict["ubound"] + 273.15 #293

    # Solar Gains
    m.phi_ia = en.Param(m.Time, initialize=dict_sg["phi_ia"])
    m.phi_st = en.Param(m.Time, initialize=dict_sg["phi_st"])
    m.phi_m = en.Param(m.Time, initialize=dict_sg["phi_m"])
    
    # PV paramaters
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])

    # Heating parameters 
    m.theta_air_outdoor = en.Param(m.Time, initialize=dict_sg["theta_out_K"])
    m.theta_air_outdoor_degC = en.Param(m.Time, initialize=dict_sg["theta_out_degC"])

    m.theta_air_indoor = en.Var(m.tm,
                                bounds=(session_dict["lbound"] + 273.15, session_dict["ubound"] + 273.15), #cooling test
                                initialize=m.T_init
                               )

    m.Q_hp_sh = en.Var(m.tm, bounds=(-20_000, 0), initialize=0) #W
    
    m.z = en.Var(domain=en.NonNegativeReals) #zzzzz
    m.min_max = en.Constraint(m.tm, rule=min_max_constraint) #zzzzz
    
    m.pv_power = en.Param(m.Time, initialize=dict_sg["pv_power"])
    m.match = en.Var(m.Time, domain=en.NonNegativeReals)
    m.M_positive = en.Var(m.Time, domain=en.NonNegativeReals)  #Excess demand
    m.M_negative = en.Var(m.Time, domain=en.NonNegativeReals)  #Excess supply
    m.supply_constraint = en.Constraint(m.Time, rule=match_constraint_bau) 
    m.mismatch_constraint = en.Constraint(m.Time, rule=mismatch_constraint_bau)
    m.rc_model_internal_temperature_r = en.Constraint(m.Time, rule=rc_simulation_temperature_constraint)
 
    # Objective Function and constraints
    m.match_pv_cooling = en.Objective(rule=objective_function_bau, sense=en.minimize)

    opt = SolverFactory('glpk') #cbc
    results = opt.solve(m, tee=True,symbolic_solver_labels=True)
    #m.load(results)    
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        #print("*************Using regular solve*************")
        results = results
    else:
        print("Infeasible problem")
        pass
    
    results_table = pd.DataFrame({
        'Q_C_nd_BAU_sim': [-1*(m.Q_hp_sh[v].value) for v in m.Q_hp_sh][1:],
    },
    index=df_sg.index)
    
    return results_table

#------------------------------------------------------------------------------
# 3. BAU simulation
#------------------------------------------------------------------------------

def bau_simulation():
    
    df = solar_gains()
    
    # Set up date/time and list of inputs/outputs for calculaton
    start_date = dt.datetime(2020,8,session_dict["start_day"],0) #DATE-X
    end_date = dt.datetime(2020,8,session_dict["end_day"]-1,23) #DATE-X
    delta = dt.timedelta(hours=1)
    time_length = end_date - start_date
    cols_g = ['hour_of_day','day_of_week','t_outside_t2m','t_outside_d2m','humidity','surface_pressure','ghi','dni','dhi','solar_position','relative_airmass','extra_radiation','solar_altitude','solar_altitude_radians','solar_azimuth','solar_azimuth_radians','solar_zenith','solar_zenith_radians','total_irradiance','phi_int','phi_sol','Q_HC_nd','Q_H_nd','Q_C_nd','theta_air','t_outside_t2m_degC','theta_air_ac', 'i_sol', 'theta_air_0', 'theta_air_10', 'theta_m_tp', 'theta_m_t', 'phi_ia', 'phi_st', 'phi_m', 'pv_power']
    n_timesteps = int(time_length.total_seconds() / delta.total_seconds())
    timestamp_list = [(start_date + x*delta) for x in range(0,n_timesteps+1)]
    
    
    current_timestamp = start_date
    current_index = 0
    
    C_m = session_dict["C_m"]
    H_tr_is = session_dict["H_tr_is"]
    H_ve_adj = session_dict["H_ve_adj"]
    H_tr_1 = session_dict["H_tr_1"]
    H_tr_w = session_dict["H_tr_w"]
    H_tr_ms = session_dict["H_tr_ms"]
    H_tr_em = session_dict["H_tr_em"]
    A_f = session_dict["A_f"]
    eta_v = param_dict["eta_v"]
    #eta_v = .87
    
    df["theta_air_0"] = ""
    df["theta_air_ac"] = ""
    df["theta_air"] = ""
    df["theta_air_10"] = ""
    df["Q_HC_nd"] = ""
    df["Q_H_nd"] = ""
    df["Q_C_nd"] = ""
    df["theta_m_t"] = ""
    
    # Initial theta_m_t value
    theta_m_t = 20
    # Initial theta_air
    theta_air_ac = 20
    theta_air = theta_air_ac
    theta_m_tp_list = []
    theta_m_tp_list.append(theta_m_t) #EJW mtp
    t_set_max = session_dict["ubound"]
    t_set_min = session_dict["lbound"]
    # Set temperatures for heating and cooling
    theta_int_C_set = t_set_max
    theta_int_H_set = t_set_min
    # Maximum heating and cooling power EJW OK?
    phi_H_max = float("inf")
    phi_C_max = -float("inf")
    
    cooling_power = 10
    
    #BROKEN!
    
    while current_timestamp <= end_date:
        
        phi_m = df.phi_m[current_index]
        phi_st = df.phi_st[current_index]
        phi_ia = df.phi_ia[current_index]
        phi_int = df.phi_int[current_index]
        phi_sol = df.phi_sol[current_index]
        
        theta_e = df.theta_out_degC[current_index]
        
        # STEP 1
        # ------ 
        # Check if heating or cooling is needed:
        phi_HC_nd_0 = 0
        H_tr_2_0 = H_tr_1 + H_tr_w
        H_tr_3_0 = 1 / (1/H_tr_2_0 + 1/H_tr_ms)
        if current_index == 0:
            theta_m_tp_0 = 0
            theta_m_t_0 = 0
        else:
            theta_m_tp_0 = theta_m_t
        theta_sup = eta_v * theta_m_tp_0 + (1 - eta_v) * theta_e #simfix
        phi_mtot_0 = phi_m + H_tr_em * theta_e + H_tr_3_0 * (phi_st + H_tr_w * theta_e + H_tr_1 * (((phi_ia + phi_HC_nd_0)/H_ve_adj) + theta_sup)) / H_tr_2_0 #simfix
        theta_m_tp_0 = theta_m_tp_list[0] #EJW mtp
        theta_m_t_0 = (theta_m_tp_0 * ((C_m/3600) - 0.5 * (H_tr_3_0 + H_tr_em)) + phi_mtot_0) / ((C_m/3600) + 0.5 * (H_tr_3_0 + H_tr_em))
        theta_m_0 = (theta_m_t_0 + theta_m_tp_0)/2
        theta_s_0 = (H_tr_ms * theta_m_0 + phi_st + H_tr_w * theta_e + H_tr_1 * (theta_sup + (phi_ia + phi_HC_nd_0)/H_ve_adj)) / (H_tr_ms + H_tr_w + H_tr_1)
        df.theta_air_0[current_index] = (H_tr_is * theta_s_0 + H_ve_adj * theta_sup + phi_ia + phi_HC_nd_0) / (H_tr_is + H_ve_adj)
        theta_air_0 = float(df.theta_air_0[current_index])
        theta_op_0 = 0.3 * theta_air_0 + 0.7 * theta_s_0
    
        if (float(theta_air_0) >= theta_int_H_set) & (float(theta_air_0) <= theta_int_C_set):
            phi_HC_nd_ac = 0
            df.theta_air_ac[current_index] = float(theta_air_0)
            df.theta_air[current_index] = df.theta_air_ac[current_index]
            df.Q_HC_nd[current_index] = 0
            df.Q_H_nd[current_index] = 0
            df.Q_C_nd[current_index] = 0
            df.theta_m_t[current_index] = theta_m_t_0
            theta_m_t = theta_m_t_0
            theta_m_tp_list.clear() #EJW mtp
            theta_m_tp_list.append(theta_m_t) #EJW mtp
            current_timestamp += delta
            current_index += 1
            continue
        else:
            pass
    
        # STEP 2
        # ------ 
        if float(theta_air_0) > theta_int_C_set:
            theta_air_set = theta_int_C_set
        elif float(theta_air_0) < theta_int_H_set:
            theta_air_set = theta_int_H_set
            
        # Apply heating factor of 10 W/m2:
        phi_HC_nd_10 = A_f * cooling_power
        H_tr_2_10 = H_tr_1 + H_tr_w
        H_tr_3_10 = 1 / (1/H_tr_2_10 + 1/H_tr_ms)
        if current_index == 0:
            theta_m_tp_10 = 0
            theta_m_t_10 = 0
        else:
            theta_m_tp_10 = theta_m_t
        theta_sup = eta_v * theta_m_tp_10 + (1 - eta_v) * theta_e #simfix
        #phi_mtot_10 = phi_m + H_tr_em * theta_sup + H_tr_3_10 * (phi_st + H_tr_w * theta_sup + H_tr_1 * (((phi_ia + phi_HC_nd_10)/H_ve_adj) + theta_sup)) / H_tr_2_10
        phi_mtot_10 = phi_m + H_tr_em * theta_e + H_tr_3_10 * (phi_st + H_tr_w * theta_e + H_tr_1 * (((phi_ia + phi_HC_nd_10)/H_ve_adj) + theta_sup)) / H_tr_2_10 #simfix
        theta_m_tp_10 = theta_m_tp_list[0] #EJW mtp
        theta_m_t_10 = (theta_m_tp_10 * ((C_m/3600) - 0.5 * (H_tr_3_10 + H_tr_em)) + phi_mtot_10) / ((C_m/3600) + 0.5 * (H_tr_3_10 + H_tr_em))
        theta_m_10 = (theta_m_t_10 + theta_m_tp_10)/2
        theta_s_10 = (H_tr_ms * theta_m_10 + phi_st + H_tr_w * theta_e + H_tr_1 * (theta_sup + (phi_ia + phi_HC_nd_10)/H_ve_adj)) / (H_tr_ms + H_tr_w + H_tr_1)
        df.theta_air_10[current_index] = (H_tr_is * theta_s_10 + H_ve_adj * theta_sup + phi_ia + phi_HC_nd_10) / (H_tr_is + H_ve_adj)
        theta_air_10 = float(df.theta_air_10[current_index])
        theta_op_10 = 0.3 * theta_air_10 + 0.7 * theta_s_10
        
        # Unrestricted heating/cooling, phi_HC_nd_un, is positive for heating and negative for cooling 
        phi_HC_nd_un = (phi_HC_nd_10*(theta_air_set - theta_air_0))/(theta_air_10 - theta_air_0)
            
        # STEP 3
        # ------    
        if (float(phi_HC_nd_un) > phi_C_max) & (float(phi_HC_nd_un) < phi_H_max):
            phi_HC_nd_ac = float(phi_HC_nd_un)
            df.theta_air_ac[current_index] = theta_air_set
            df.theta_air[current_index] = df.theta_air_ac[current_index]
            # The energy need (MJ) for heating or cooling for a given hour, Q_HC_nd, is positive in the case of heating need and negative in the case of cooling need
            df.Q_HC_nd[current_index] = phi_HC_nd_ac * .1#0.036
            df.Q_H_nd[current_index] = max(0, float(df.Q_HC_nd[current_index]))
            df.Q_C_nd[current_index] = abs(min(0, float(df.Q_HC_nd[current_index])))
            df.theta_m_t[current_index] = theta_m_t_10
            theta_m_t = theta_m_t_10
            theta_m_tp_list.clear() #EJW mtp
            theta_m_tp_list.append(theta_m_t) #EJW mtp
            current_timestamp += delta
            current_index += 1
            #print('mark5')
            #print('<<<End calculation for timestep, Case 1 or 5>>>')
            continue
        
        # STEP 4
        # ------ 
        else:
            #print('mark6')
            if float(phi_HC_nd_un) > 0:
                phi_HC_nd_ac = session_dict["lbound"]
                #print('mark7')
            elif float(phi_HC_nd_un) < 0:
                phi_HC_nd_ac = session_dict["ubound"]
                #print('mark8')
        # Other combined heat conductances
        H_tr_2 = H_tr_1 + H_tr_w
        H_tr_3 = 1 / (1/H_tr_2 + 1/H_tr_ms)
        # Set theta_m_tp as theta_m_t from previous step
        if current_index == 0:
            theta_m_tp = 0
            theta_m_t = 0
        else:
            theta_m_tp = theta_m_t
        theta_sup = eta_v * theta_m_tp + (1 - eta_v) * theta_e #simfix
        phi_HC_nd = phi_HC_nd_ac    
        #phi_mtot = phi_m + H_tr_em * theta_sup + H_tr_3 * (phi_st + H_tr_w * theta_sup + H_tr_1 * (((phi_ia + phi_HC_nd)/H_ve_adj) + theta_sup)) / H_tr_2
        phi_mtot = phi_m + H_tr_em * theta_e + H_tr_3 * (phi_st + H_tr_w * theta_e + H_tr_1 * (((phi_ia + phi_HC_nd)/H_ve_adj) + theta_sup)) / H_tr_2 #simfix
        theta_m_tp = theta_m_tp_list[0] #EJW mtp
        df.theta_m_t[current_index] = (theta_m_tp * ((C_m/3600) - 0.5 * (H_tr_3 + H_tr_em)) + phi_mtot) / ((C_m/3600) + 0.5 * (H_tr_3 + H_tr_em))
        theta_m_t = df.theta_m_t[current_index]
        theta_m_tp_list.clear() #EJW mtp
        theta_m_tp_list.append(theta_m_t) #EJW mtp
        theta_m = (theta_m_t + theta_m_tp)/2 
        theta_s = (H_tr_ms * theta_m + phi_st + H_tr_w * theta_e + H_tr_1 * (theta_sup + (phi_ia + phi_HC_nd)/H_ve_adj)) / (H_tr_ms + H_tr_w + H_tr_1)
        df.theta_air_ac[current_index] = (H_tr_is * theta_s + H_ve_adj * theta_sup + phi_ia + phi_HC_nd) / (H_tr_is + H_ve_adj)
        theta_air_ac = float(df.theta_air_ac[current_index])
        df.theta_air[current_index] = df.theta_air_ac[current_index]
        theta_op = 0.3 * theta_air_ac + 0.7 * theta_s
        # The energy need (MJ) for heating or cooling for a given hour, Q_HC_nd, is positive in the case of heating need and negative in the case of cooling need
        df.Q_HC_nd[current_index] = phi_HC_nd_ac * .1#0.036
        df.Q_HC_nd[current_index] = phi_HC_nd_ac / 277.78 #simfix
        # Heating
        df.Q_H_nd[current_index] = max(0, float(df.Q_HC_nd[current_index])) 
        # Cooling
        df.Q_C_nd[current_index] =  abs(min(0, float(df.Q_HC_nd[current_index]))) 
        
    results_lite = {"Q_C_nd_BAU_sim":df.Q_C_nd, "theta_air_indoor_BAU_sim":df.theta_air_ac}
    df_results_bau = pd.DataFrame(results_lite)
    #df_results_bau.to_csv('Output/cm6_results_BAU_testing.csv')
    
    return df_results_bau


#------------------------------------------------------------------------------
# 4. Outputs/KPIs 
#------------------------------------------------------------------------------

def graph_output():
    
    results_table_dr = opt_model()
    results_table_sim = bau_simulation() #opt_model_bau()
    results_table = pd.merge(results_table_dr, results_table_sim, left_index=True, right_index=True)
    results_table = results_table.rename(columns={"Q_hp_sh":"Q_hp_sh_dr"})
    results_table['Q_hp_sh_dr_total'] = results_table['Q_hp_sh_dr'] - results_table['pv_power'] 
    results_table['Q_hp_sh_dr_actual'] = np.where(results_table['pv_power'] > results_table['Q_hp_sh_dr_total'], results_table['pv_power'] , results_table['Q_hp_sh_dr_total'])
    
    #results_table.to_csv('Output/CM6_RESULTS.csv')    
    #Cutoff first day
    results_table = results_table[24:len(results_table)]
    ax = results_table.Q_C_nd_BAU_sim.plot(color='black', label='CD - BAU')
    ax = results_table.Q_hp_sh_dr.plot(color='blue',  
                                    label='CD - DR')
                                    #label='Cooling Demand - DR [total] (W)')
    if session_dict["scenario"] == "PV Match":
        ax = results_table.pv_power.plot(color='red', label='PVS')
    #ax = results_table.matched.plot(label='Cooling Demand - DR [PVSC] (W)', color='blue', linestyle='dashed')
    #ax = results_table.Q_hp_sh_dr_total.plot(label='Cooling Demand - DR [grid] (W)', color='grey')
    ax.set_ylabel('Power (W)')
    ax.set_title(session_dict["country_name"]+': '+session_dict["scenario"]+' scenario vs. BAU simulation')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=3)
    plt.tight_layout()
    fig_name_g1 = 'Output/' + '_7days.png'
    #plt.savefig(fig_name_g1)
    plt.show() #delete
    plt.clf()
  

    return fig_name_g1

def load_graphics(country_name):
    session_dict = load_parameters(country_name)
    results_table_dr = opt_model()
    results_table_sim = bau_simulation() #opt_model_bau()
    results_table = pd.merge(results_table_dr, results_table_sim, left_index=True, right_index=True)
    results_table = results_table.rename(columns={"Q_hp_sh":"Q_hp_sh_dr"})
    results_table['Q_hp_sh_dr_total'] = results_table['Q_hp_sh_dr'] - results_table['pv_power'] 
    results_table['Q_hp_sh_dr_actual'] = np.where(results_table['pv_power'] > results_table['Q_hp_sh_dr_total'], results_table['pv_power'] , results_table['Q_hp_sh_dr_total'])
    results_table = results_table[24:len(results_table)]
    
    graphics =  [
        {
            'type': 'line',
            'label': session_dict["country_name"]+': '+session_dict["scenario"]+' scenario vs. BAU simulation',
            'xlabel': 'Power (W)',
            'ylabel': 'Hour',
            # 'options': {
            #     'scales': {
            #         'xAxes': [{'stacked': True}],  # Enable stacking on the x-axis
            #         'yAxes': [{'stacked': True}]  # Enable stacking on the y-axis
            #     }
            # },
            "data": {
                "labels": ['2030', '2040', '2050'],  # Labels for x-axis
                "datasets": [
                    {
                        "label": "Cooling Demand - BAU",
                        "backgroundColor": "#2b2a30",  # Color for the bottom layer
                        "data": [results_table['Q_C_nd_BAU_sim']]  # Data for each x-tick: baseline, moderate, high
                    },
                    {
                        "label": "Cooling Demand - DR",
                        "backgroundColor":  "#65fe60",  # Color for the bottom layer
                        "data": [results_table['Q_hp_sh_dr']]  # Data for each x-tick: baseline, moderate, high
                    },
                    {
                        "label": "PV Supply",
                        "backgroundColor":  "#fe6560",  # Color for the bottom layer
                        "data": [results_table['pv_power']]  # Data for each x-tick: baseline, moderate, high
                    }
                ]
            }
        }
    ]
    return graphics
#graph_output()

