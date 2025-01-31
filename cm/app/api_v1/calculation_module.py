import os

from osgeo import gdal

from ..helper import generate_output_file_tif, create_zip_shapefiles
from ..constant import CM_NAME
import pandas as pd
import time
import logging

from .my_calculation_module_directory.COOLLIFE_CM5_v1 import load_graphics
#from .my_calculation_module_directory.extract_nuts_id import get_country_name

""" Entry point of the calculation module function"""

#TODO: CM provider must "change this code"
#TODO: CM provider must "not change input_raster_selection,output_raster  1 raster input => 1 raster output"
#TODO: CM provider can "add all the parameters he needs to run his CM
#TODO: CM provider can "return as many indicators as he wants"
def calculation(output_directory, inputs_raster_selection, inputs_parameter_selection): 
                #areas):
    #TODO the folowing code must be changed by the code of the calculation module

    # List of valid country codes (FULL LIST -> CM Developper must select the country they need and at them in the next list)
    #validatecountrycode = {
    #    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czech Republic",
    #    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France", "EL": "Greece",
    #    "HU": "Hungary", "HR": "Croatia", "IE": "Ireland", "IT": "Italy", "LT": "Lithuania",
    #    "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta", "NL": "Netherlands", "PL": "Poland",
    #    "PT": "Portugal", "RO": "Romania", "ES": "Spain", "SE": "Sweden", "SI": "Slovenia",
    #    "SK": "Slovakia", "UK": "United Kingdom", "AL": "Albania", "ME": "Montenegro",
    #    "MK": "North Macedonia", "RS": "Serbia", "TR": "Turkey", "CH": "Switzerland", "IS": "Iceland",
    #    "LI": "Liechtenstein", "NO": "Norway"
    #}

    '''
    validatecountrycode = {
        "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus", "CZ": "Czech Republic", "DK": "Denmark", 
        "EE": "Estonia", "FI": "Finland", "FR": "France", "DE": "Germany", "EL": "Greece", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
        "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
        "SK": "Slovakia", "SI": "Slovenia", "ES": "Spain", "SE": "Sweden"
    }

    errorCode = 0

    # Check if areas is empty
    if not areas:
        errorCode = 1
    # Check if areas contains more than one element
    elif len(areas) > 1:
        errorCode = 2
    # Check if the code in areas is valid
    elif areas[0] not in validatecountrycode:
        errorCode = 3

    # If errorCode is not zero, return an error
    if errorCode != 0:
        result['name'] = CM_NAME
        result['indicator'] = []

        if errorCode == 1:
            error_message = "No area selected. Please select one area."
        elif errorCode == 2:
            error_message = "More than one area selected. Please select only one area."
        elif errorCode == 3:
            country_list = ', '.join(validatecountrycode.values())
            error_message = (
                f"The selected country is not supported! Please select one of the following countries: {country_list}."
            )

        # Set indicator with error format
        # ! The result must be check to be sure the format is correct for your CM.
        result['indicator'] = [
            {
                "name": error_message,
                "value": "",
                "unit": ""
            }
        ]

        result['graphics'] = []
        result['vector_layers'] = []
        result['raster_layers'] = []

        print('Result:', result)
        return result


    country_name = validatecountrycode.get(areas[0], "Unknown Country")
    '''
    
    country_name = inputs_parameter_selection['country_name']
    graphics = load_graphics(country_name)
    
    #path2data = os.path.join(os.path.dirname(__file__), 'data')
    #file_path_nuts_code = os.path.join(path2data, "nuts_id_number.csv")
    #country_name = get_country_name(file_path_nuts_code,path_nuts_id_tif)
    
    # generate the output raster file
    #output_raster1 = generate_output_file_tif(output_directory)
    # retrieve the inputs all input defined in the signature
    #factor = float(inputs_parameter_selection["multiplication_factor"])

    #retrieve the inputs layes
    #input_raster_selection =  inputs_raster_selection["heat"]


    #retrieve the inputs layes
    """
        print("inputs_vector_selection ",inputs_vector_selection)
        a_vehicle_stock =  inputs_vector_selection["a_vehicle_stock"]
        print("a_vehicle_stock ",a_vehicle_stock)
        b_final_energy_consumption =  inputs_vector_selection["b_final_energy_consumption"]
        print("b_final_energy_consumption ",b_final_energy_consumption)
        b_vehicle_stock =  inputs_vector_selection["b_vehicle_stock"]
        print("b_vehicle_stock ",b_vehicle_stock)
        bau_final_energy_consumption =  inputs_vector_selection["bau_final_energy_consumption"]
        print("bau_final_energy_consumption ",bau_final_energy_consumption)"""

    # TEST FOR VECTOR



    """
    # TODO this part bellow must be change by the CM provider
    ds = gdal.Open(input_raster_selection)
    ds_band = ds.GetRasterBand(1)

    #----------------------------------------------------
    pixel_values = ds.ReadAsArray()
    #----------Reduction factor----------------

    pixel_values_modified = pixel_values* float(factor)
    hdm_sum  = float(pixel_values_modified.sum())/1000


    gtiff_driver = gdal.GetDriverByName('GTiff')
    #print ()
    out_ds = gtiff_driver.Create(output_raster1, ds_band.XSize, ds_band.YSize, 1, gdal.GDT_UInt16, ['compress=DEFLATE',
                                                                                                         'TILED=YES',
                                                                                                         'TFW=YES',
                                                                                                         'ZLEVEL=9',
                                                                                                         'PREDICTOR=1'])
    out_ds.SetProjection(ds.GetProjection())
    out_ds.SetGeoTransform(ds.GetGeoTransform())

    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0,0,0,255))
    ct.SetColorEntry(1, (110,220,110,255))
    out_ds.GetRasterBand(1).SetColorTable(ct)

    out_ds_band = out_ds.GetRasterBand(1)
    out_ds_band.SetNoDataValue(0)
    out_ds_band.WriteArray(pixel_values_modified)

    del out_ds
    # output geneneration of the output
    graphics = []
    vector_layers = []
    """

    #TODO to create zip from shapefile use create_zip_shapefiles from the helper before sending result
    #TODO exemple  output_shpapefile_zipped = create_zip_shapefiles(output_directory, output_shpapefile)
    result = dict()
    result['name'] = CM_NAME
    #result['indicator'] = [
    #    {"unit": "GWh", "name": "Heat density total multiplied by  {}".format(factor),"value": str(hdm_sum)}
    #]
    result['graphics'] = graphics
    #result['vector_layers'] = vector_layers
    #result['raster_layers'] = [{"name": "layers of heat_densiy {}".format(factor),"path": output_raster1, "type": "heat"}]
    print ('result',result)
    return result


def colorizeMyOutputRaster(out_ds):
    ct = gdal.ColorTable()
    ct.SetColorEntry(0, (0,0,0,255))
    ct.SetColorEntry(1, (110,220,110,255))
    out_ds.SetColorTable(ct)
    return out_ds
