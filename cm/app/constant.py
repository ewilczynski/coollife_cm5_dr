
CELERY_BROKER_URL_DOCKER = 'amqp://admin:mypass@rabbit:5672/'
CELERY_BROKER_URL_LOCAL = 'amqp://localhost/'


CM_REGISTER_Q = 'rpc_queue_CM_register' # Do no change this value

CM_NAME = 'CM - Scale heat and cool density maps'
RPC_CM_ALIVE= 'rpc_queue_CM_ALIVE' # Do no change this value
RPC_Q = 'rpc_queue_CM_compute' # Do no change this value
CM_ID = 1 # CM_ID is defined by EASILab (support@easilabdev.ch)
PORT_LOCAL = int('500' + str(CM_ID))
PORT_DOCKER = 80

#TODO ********************setup this URL depending on which version you are running***************************

CELERY_BROKER_URL = CELERY_BROKER_URL_DOCKER
PORT = PORT_DOCKER

#TODO ********************setup this URL depending on which version you are running***************************

TRANFER_PROTOCOLE ='http://'
INPUTS_CALCULATION_MODULE = [
    {'input_name': 'Multiplication factor',
     'input_type': 'input',
     'input_parameter_name': 'multiplication_factor',
     'input_value': '1',
     'input_priority': 0,
     'input_unit': '',
     'input_min': 0,
     'input_max': 10, 'cm_id': CM_ID  # Do no change this value
     },
]


SIGNATURE = {

    "category": "Demand",
    "cm_name": CM_NAME,
    "wiki_url": "https://wiki.hotmaps.hevs.ch/en/CM-Scale-heat-and-cool-density-maps",
    "layers_needed": [],
    "type_layer_needed": [
        {"type": "heat", "description": "Choose a heat demand density layer."}
    ],
    "type_vectors_needed": [],
    "cm_url": "Do not add something",
    "cm_description": "This calculation module allows to scale the heat demand density layer up or down.",
    "cm_id": CM_ID,
    'inputs_calculation_module': INPUTS_CALCULATION_MODULE
}
