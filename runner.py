import os
import sys
import json
import argparse

import pprint
from tqdm import tqdm
from p_tqdm import p_map
from histoprint import print_hist
import numpy as np

import uproot
from coffea import hist
from coffea.nanoevents import NanoEventsFactory
from coffea.util import load, save

from boostedhiggs.hbbprocessor import HbbProcessor

from coffea import processor

def validate(file):
    try:
        fin = uproot.open(file)
        return fin['Events'].numentries
    except:
        print("Corrupted file: {}".format(file))
        return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run analysis on baconbits files using processor coffea files')
    parser.add_argument('--output', default=r'hists.coffea', help='Output histogram filename (default: %(default)s)')
    parser.add_argument('--samplejson', default='metadata/dataset_local.json', help='JSON file containing dataset and file locations (default: %(default)s)')
    parser.add_argument('--sample', default='test_skim', help='The sample to use in the sample JSON (default: %(default)s)')
    parser.add_argument('--limit', type=int, default=None, metavar='N', help='Limit to the first N files of each dataset in sample JSON')
    parser.add_argument('--chunk', type=int, default=500000, metavar='N', help='Number of events per process chunk')
    parser.add_argument('--max', type=int, default=None, metavar='N', help='Max number of chunks to run in total')
    parser.add_argument('--validate', action='store_true', help='Do not process, just check all files are accessible')
    parser.add_argument('--only', type=str, default=None, help='Only process specific dataset or file')
    parser.add_argument('--executor', choices=['iterative', 'futures', 'parsl', 'uproot'], default='uproot', help='The type of executor to use (default: %(default)s)')
    parser.add_argument('-j', '--workers', type=int, default=12, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
    args = parser.parse_args()
    if args.output == parser.get_default('output'):
        args.output = 'hists_%s.coffea' % args.sample


    # load dataset
    with open(args.samplejson) as f:
        sample_dict = json.load(f)
    
    for key in sample_dict.keys():
        sample_dict[key] = sample_dict[key][:args.limit]
    # pprint.pprint(list(sample_dict.keys()))

    # For debugging
    if args.only is not None:
        if args.only in sample_dict.keys():  # is dataset
            sample_dict = dict([(args.only, sample_dict[args.only])])
        if "*" in args.only: # wildcard for datasets
            _new_dict = {}
            print("Will only proces the following datasets:")
            for k, v in sample_dict.items():
                if k.lstrip("/").startswith(args.only.rstrip("*")):
                    print("    ", k)
                    _new_dict[k] = v
            sample_dict = _new_dict
        else:  # is file
            for key in sample_dict.keys():
                if args.only in sample_dict[key]:
                    sample_dict = dict([(key, [args.only])]) 


    # Scan if files can be opened
    if args.validate:
        for sample in sample_dict.keys():
            r = p_map(validate, sample_dict[sample], num_cpus=args.workers,
                      desc=f'Validating {sample[:20]}...')
            print("Events:", np.sum(list(r)))
        sys.exit(0)
    
    
    if args.executor in ['uproot', 'iterative']:
        if args.executor == 'iterative':
            _exec = processor.iterative_executor
        else:
            _exec = processor.futures_executor
        output = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=HbbProcessor(),
                                    executor=_exec,
                                    executor_args={
                                        'skipbadfiles':True,
                                        'schema': processor.NanoAODSchema, 
                                        'flatten':True, 
                                        'workers': 4},
                                    chunksize=args.chunk, maxchunks=args.max
                                    )
    elif args.executor == 'parsl':
        from cfg_parsl import *
        _exec = processor.parsl_executor
        output, metrics = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=HbbProcessor(),
                                    executor=_exec,
                                    executor_args={
                                        'skipbadfiles':True,
                                        'savemetrics':True,
                                        'schema': processor.NanoAODSchema, 
                                        'flatten':True, 
                                        'config': None},
                                    chunksize=args.chunk, maxchunks=args.max
                                    )


    save(output, args.output)
    ### VIZ
    def show_hists(h, title=''):
        hall = []
        names = []
        for name in h.axis(h.sparse_axes()[0]).identifiers():
            _h = h[name].project('msd').values()[()]
            bins =  h[name].axis('msd').edges()
            hall.append(_h)
            names.append(name.name)
            
        print_hist((hall, bins), title=title, 
                labels=names,
                stack=True,
                columns=100,
                bg_colors="rgbcmy")

    sr_pre = (output['templates'].integrate('region', 'signal')
                .integrate('systematic', 'nominal')
                .integrate('ddb')
                .integrate('pt')
                .integrate('ddc', slice(0.83,None))
    )

    # Sort and group samples
    from collections import OrderedDict
    import string
    short = [k.split("_")[0].lstrip('/') for k in list(sample_dict.keys())]
    mapper = dict(zip(sample_dict.keys(), short))

    alphabet = "QTZWG" + string.ascii_uppercase + string.ascii_lowercase
    groups = list(set(mapper.values()))
    map_dict = OrderedDict()
    for group in sorted(groups, key=lambda word: [alphabet.index(c) for c in word]):
        map_dict[group] = [k for k, v in mapper.items() if v == group]

    _cat = hist.Cat("dataset", "Process", sorting='placement')
    sr_pre = sr_pre.group('dataset', _cat, map_dict)
    
    sr = sr_pre.integrate('genflavor')
    src = sr_pre.integrate('genflavor', output['templates'].identifiers('genflavor')[-2])
    
    show_hists(sr.project('msd', 'dataset'), title="Samples (SR)")
    try:
        show_hists(src.project('msd', 'dataset'), title="Samples (SR-charm)")
    except:
        pass