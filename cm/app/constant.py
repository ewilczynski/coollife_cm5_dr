
CELERY_BROKER_URL_DOCKER = 'amqp://admin:mypass@rabbit:5672/'
CELERY_BROKER_URL_LOCAL = 'amqp://localhost/'


CM_REGISTER_Q = 'rpc_queue_CM_register' # Do no change this value

CM_NAME = 'CM - Demand-side Management/Demand Response'
RPC_CM_ALIVE= 'rpc_queue_CM_ALIVE' # Do no change this value
RPC_Q = 'rpc_queue_CM_compute' # Do no change this value
<<<<<<< HEAD
CM_ID = 21 # CM_ID is defined by the enegy research center of Martigny (CREM)
=======
CM_ID = 1 # CM_ID is defined by EASILab (citiwatts@hevs.ch)
>>>>>>> b6ca004f2a18c142b77a61cdb219226b2ee125cb
PORT_LOCAL = int('500' + str(CM_ID))
PORT_DOCKER = 80

#TODO ********************setup this URL depending on which version you are running***************************

CELERY_BROKER_URL = CELERY_BROKER_URL_DOCKER
PORT = PORT_DOCKER

#TODO ********************setup this URL depending on which version you are running***************************

TRANFER_PROTOCOLE ='http://'
INPUTS_CALCULATION_MODULE = [
    #{'input_name': 'Multiplication factor',
    # 'input_type': 'input',
    # 'input_parameter_name': 'multiplication_factor',
    # 'input_value': '1',
    # 'input_priority': 0,
    # 'input_unit': '',
    # 'input_min': 0,
    # 'input_max': 10, 'cm_id': CM_ID  # Do no change this value
    # },
    {'input_name': 'Country Name',
     'input_type': 'select',
     'input_parameter_name': 'country_name',
     'input_value': ["Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"],
     'input_priority': 0,
     'input_unit': '',
     'input_min': 'none',
     'input_max': 'none', 'cm_id': CM_ID  # Do no change this value
     },
]


SIGNATURE = {

    "category": "Demand",
    "cm_name": CM_NAME,
    "wiki_url": "https://coollife-project.github.io/wiki/cm-5-demand-side-managementdemand-response/",
    "layers_needed": [],
    "type_layer_needed": [
        {"type": "heat", "description": "Choose a heat demand density layer."}
    ],
    "type_vectors_needed": [],
    "cm_url": "Do not add something",
    "cm_description": "This calculation module calculates potential to match cooling demand with PV supply.",
    "cm_id": CM_ID,
    'inputs_calculation_module': INPUTS_CALCULATION_MODULE
}
