
import argparse
from typing import Dict
from utils import OptionBridge, registry_values

def create_argument_parser(opts: Dict[str, OptionBridge]) -> argparse.ArgumentParser :
    parser = argparse.ArgumentParser()
    for key, value in opts.items():
        group = parser.add_argument_group(key)
        for k,v in value.items():
            if not v.required:
                group.add_argument(f"--{k}", default=v(), type=type(v.value), 
                                   help=v.desc, 
                                   choices=None if not v.choices else v.choices)
            else:
                group.add_argument(f"--{k}", type=type(v.value), help=v.desc, 
                                   required=v.required,
                                   choices=None if not v.choices else v.choices)
    return parser



def get_arg_groups(parser, opts):
    arg_groups = {}

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(opts, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    return arg_groups
