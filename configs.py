import argparse
import shared.misc_utils as utils 
import os
from task_specific_params import task_lst

def str2bool(v):
    return v.lower() in ('true', '1')

def initialize_task_settings(args,task):

    try:
        task_params = task_lst[task]
    except:
        raise Exception('Task is not implemented.') 

    for name, value in task_params._asdict().items():
    	args[name] = value


    # args['task_name'] = task_params.task_name
    # args['input_dim'] = task_params.input_dim
    # args['n_nodes'] = task_params.n_nodes
    # if args['decode_len'] == None:
    #     args['decode_len'] = task_params.decode_len

    return args