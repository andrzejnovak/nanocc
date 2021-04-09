import os
import sys
import json
import argparse
import time

import pprint
from tqdm import tqdm
# from p_tqdm import p_map
from histoprint import print_hist
import numpy as np

import uproot
from coffea import hist
from coffea.nanoevents import NanoEventsFactory
from coffea.util import load, save

from boostedhiggs.hbbprocessor import HbbProcessor

from coffea import processor

def validate(fname):
    try:
        with uproot.open(fname) as fin:
            return fin['Events'].num_entries
    except:
        print("Corrupted file: {}".format(fname))
        return fname
        
### VIZ
def show_hists(h, title='', scaleH=False):
    hall = []
    names = []
    for name in h.axis(h.sparse_axes()[0]).identifiers():
        _h = h[name].project('msd').values()[()]
        if name.name.startswith("GluGlu") and scaleH:
            if np.isreal(scaleH):
                _h *= scaleH
            else:
                _h *= 100
        bins =  h[name].axis('msd').edges()
        hall.append(_h)
        names.append(name.name)
        
    print_hist((hall, bins), title=title, 
            labels=names,
            stack=True,
            symbols=" ",
            #columns=100,
            bg_colors="rgbcmy")


if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Run analysis on baconbits files using processor coffea files')
    parser.add_argument('--id', '--identifier', dest='identifier', default=r'hists', help='File identifier to carry through (default: %(default)s)')
    parser.add_argument('--output', default=r'hists.coffea', help='Output histogram filename (default: %(default)s)')
    parser.add_argument('--json', '--samples', '--paths', dest='samplejson', default='metadata/dataset_local.json', help='JSON file containing dataset and file locations (default: %(default)s)')
    parser.add_argument('--limit', type=int, default=None, metavar='N', help='Limit to the first N files of each dataset in sample JSON')
    parser.add_argument('--chunk', type=int, default=500000, metavar='N', help='Number of events per process chunk')
    parser.add_argument('--max', type=int, default=None, metavar='N', help='Max number of chunks to run in total')
    parser.add_argument('--validate', action='store_true', help='Do not process, just check all files are accessible')
    parser.add_argument('--v2', action='store_true', help='Use DDX v2')
    parser.add_argument('--rew', action='store_true', help='Reweight powheg sample to NNLOPS')

    parser.add_argument("--jec", type=str2bool, default='True', choices={True, False}, const=True, nargs='?', help="Tighter gen match requirements")
    parser.add_argument("--tightMatch", type=str2bool, default='True', choices={True, False}, help="Tighter gen match requirements")

    parser.add_argument("--looseTau", type=str2bool, default='False', choices={True, False}, help="Looser tau veto")
    
    parser.add_argument('--newTrigger', action='store_true', help='Use new trig map')
    parser.add_argument('--particleNet', action='store_true', help='Use ParticleNet')
    parser.add_argument('--particleNetMix', action='store_true', help='Use ParticleNet')
    parser.add_argument('--only', type=str, default=None, help='Only process specific dataset or file')
    parser.add_argument('--executor', choices=['iterative', 'futures', 'parsl', 'uproot','dask'], default='uproot', help='The type of executor to use (default: %(default)s)')
    parser.add_argument('--year', choices=['2016', '2017','2018'], required=True, help='Year to pass to the processor (triggers)')
    parser.add_argument('-j', '--workers', type=int, default=12, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
    args = parser.parse_args()
    print(args)
    if args.output == parser.get_default('output'):
        if args.identifier == parser.get_default('identifier'):
            args.output = f'hists_{args.sample}.coffea'
        else:
            args.output = f'hists_{args.identifier}.coffea'

    # load dataset
    with open(args.samplejson) as f:
        sample_dict = json.load(f)
    
    for key in sample_dict.keys():
        sample_dict[key] = sample_dict[key][:args.limit]

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
        start = time.time()
        from p_tqdm import p_map
        all_invalid = []
        for sample in sample_dict.keys():
            _rmap = p_map(validate, sample_dict[sample], num_cpus=args.workers,
                      desc=f'Validating {sample[:20]}...')
            _results = list(_rmap)
            counts = np.sum([r for r in _results if np.isreal(r)])
            all_invalid += [r for r in _results if type(r) == str]
            print("Events:", np.sum(counts))
        print("Bad files:")
        for fi in all_invalid:
            print(f"  {fi}")
        end = time.time()
        print("TIME:", time.strftime("%H:%M:%S", time.gmtime(end-start)))
        if input("Remove bad files? (y/n)") == "y": 
            print("Removing:")
            for fi in all_invalid:
                print(f"Removing: {fi}")            
                os.system(f'rm {fi}')
        sys.exit(0)
  

    processor_object = HbbProcessor(v2=args.v2, v3=args.particleNet, v4=args.particleNetMix, year=args.year,
                                    nnlops_rew=args.rew, skipJER=not args.jec, tightMatch=args.tightMatch,
                                    newTrigger=args.newTrigger, looseTau=args.looseTau,
                                    )

    if args.executor in ['uproot', 'iterative']:
        if args.executor == 'iterative':
            _exec = processor.iterative_executor
        else:
            _exec = processor.futures_executor
        output, metrics = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=processor_object,
                                    executor=_exec,
                                    executor_args={
                                        'skipbadfiles':True,
                                        'savemetrics':True,
                                        'schema': processor.NanoAODSchema, 
                                        'workers': 4},
                                    chunksize=args.chunk, maxchunks=args.max,
                                    #shuffle=True,
                                    )
        import pprint
        pprint.pprint(metrics)
    elif args.executor == 'parsl':
        import parsl
        from parsl.app.app import python_app, bash_app
        from parsl.configs.local_threads import config
        from parsl.providers import LocalProvider, CondorProvider, SlurmProvider
        from parsl.channels import LocalChannel
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor
        from parsl.launchers import SrunLauncher

        from parsl.addresses import address_by_hostname

        x509_proxy = 'x509up_u%s'%(os.getuid())
        wrk_init = '''
        export XRD_RUNFORKHANDLER=1
        export X509_USER_PROXY=/tmp/x509up_u27388
        export X509_CERT_DIR=/home/anovak/certs/
        ulimit -u 32768
        '''
        twoGB = 2048
        nproc = 16
        # sched_opts = '''
        # #SBATCH --cpus-per-task=%d
        # #SBATCH --mem-per-cpu=%d
        # ''' % (nproc, twoGB, )


        slurm_htex = Config(
            executors=[
                HighThroughputExecutor(
                    label="coffea_parsl_slurm",
                    address=address_by_hostname(),
                    prefetch_capacity=0,
                    max_workers=20,
                    #suppress_failure=True,
                    provider=SlurmProvider(
                        channel=LocalChannel(script_dir='test_parsl'),
                        launcher=SrunLauncher(),
                        max_blocks=(args.workers)+5,
                        init_blocks=args.workers, 
                        partition='all',
                        # scheduler_options=sched_opts,   # Enter scheduler_options if needed
                        worker_init=wrk_init,         # Enter worker_init if needed
                        walltime='03:00:00'
                    ),
                )
            ],
            retries=20,
        )
        dfk = parsl.load(slurm_htex)

        _exec = processor.parsl_executor
        output, metrics = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=processor_object,
                                    executor=_exec,
                                    executor_args={
                                        'skipbadfiles':True,
                                        'savemetrics':True,
                                        'schema': processor.NanoAODSchema, 
                                        # 'mmap':True,
                                        'config': None},
                                    chunksize=args.chunk, maxchunks=args.max
                                    )
        print(metrics)                                    

    elif args.executor == 'dask':
        from dask_jobqueue import SLURMCluster
        from distributed import Client
        from dask.distributed import performance_report

        cluster = SLURMCluster(
            queue='all',
            cores=40,
            processes=1,
            memory="500GB",
            retries=10,
            walltime='00:30:00',
            env_extra=['ulimit -u 32768', 'ulimit -n 8000'],
        )
        cluster.scale(jobs=args.workers)

        print(cluster.job_script())
        client = Client(cluster)
        print(cluster)
        with performance_report(filename="dask-report.html"):
            output = processor.run_uproot_job(sample_dict,
                                        treename='Events',
                                        processor_instance=processor_object,
                                        executor=processor.dask_executor,
                                        executor_args={
                                            'client': client,
                                            'skipbadfiles':True,
                                            'schema': processor.NanoAODSchema, 
                                        },
                                        chunksize=args.chunk, maxchunks=args.max
                            )
    save(output, args.output)
