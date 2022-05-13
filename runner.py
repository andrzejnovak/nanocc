import os
import sys
import json
import argparse
import time
import pprint
import pickle

from tqdm import tqdm
# from p_tqdm import p_map
from histoprint import print_hist
import numpy as np

import uproot
from coffea import hist
from coffea.nanoevents import NanoEventsFactory
from coffea.util import load, save

def ask_user(query):
    print(query)
    response = ''
    while response.lower() not in {"yes", "no"}:
        response = input("Please enter yes or no: ")
    return response.lower() == "yes"

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
    parser.add_argument("--skip", type=str2bool, default='True', choices={True, False}, help="Skip bad files")
    
    parser.add_argument('--rew', action='store_true', help='Reweight powheg sample to NNLOPS')

    parser.add_argument("--systs", type=str2bool, default='True', choices={True, False}, help="Run systematics")
    parser.add_argument('--tagger', dest='tagger', choices=['v2', 'v1', 'v3', 'v4'], default='v2')
    parser.add_argument("--jec", type=str2bool, default='True', choices={True, False}, help="Tighter gen match requirements")
    parser.add_argument("--tightMatch", type=str2bool, default='True', choices={True, False}, help="Tighter gen match requirements")
    parser.add_argument("--newvjets", type=str2bool, default='True', choices={True, False}, help="New W corrections")
    parser.add_argument("--newTrigger", type=str2bool, default='True', choices={True, False}, help="New trigger SFs")
    parser.add_argument("--skipRunB", type=str2bool, default='False', choices={True, False}, help="Skip run B of 2017 (bad triggers)")
    parser.add_argument("--finebins", type=str2bool, default='False', choices={True, False}, help="Run with 2x as many mass bins")

    parser.add_argument("--looseTau", type=str2bool, default='True', choices={True, False}, help="Looser tau veto")
    parser.add_argument('--arb', choices=['pt', 'n2', 'ddb', 'ddc'], default='ddc', help='Which jet to take')
    parser.add_argument('--tag', choices=['deepcsv', 'deepjet'], default='deepcsv', help='Which ak4 tagger to use')
    parser.add_argument("--ewkHcorr", type=str2bool, default='True', choices={True, False}, help="EWK Corrections for H signals. (Lowers yield)")

    parser.add_argument('--chunkify', action='store_true', help='chunk-chunk')
    parser.add_argument('--monitor', action='store_true', help='run parsl with monitoring')
    parser.add_argument('--merges', type=str2bool, default='True', choices={True, False}, help='Turn on merging')

    parser.add_argument('--only', type=str, default=None, help='Only process specific dataset or file')
    parser.add_argument('--executor', choices=['iterative', 'futures', 'parsl', 'uproot','dask'], default='uproot', help='The type of executor to use (default: %(default)s)')
    parser.add_argument('--dash', type=int, help='Dashboard address for dask', default=8787)
    parser.add_argument('--year', choices=['2016', '2017','2018'], required=True, help='Year to pass to the processor (triggers)')
    parser.add_argument('-j', '--workers', type=int, default=12, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
    args = parser.parse_args()
    print(args)
    if args.rew and args.year != '2018':
        print('Are you sure you want to reweight POWHEG->NNLOPS in non-2018?')
        user_input = input('Confirm? [Y/N]')
        if user_input.lower() not in ('y', 'yes'):
            sys.exit()
    elif not args.rew and args.year == '2018':
        print('Sure you dont want to reweight POWHEG->NNLOPS in 2018?')
        user_input = input('Confirm? [Y/N]')
        if user_input.lower() not in ('y', 'yes'):
            sys.exit()
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
        _new_dict = {}
        if "," in args.only:
            keep_only = args.only.split(",")
        else:
            keep_only = [args.only]
        print("Will only proces the following datasets:")
        for keep in keep_only:
            if "*" in keep: # wildcard for datasets
                for k, v in sample_dict.items():
                    if k.lstrip("/").startswith(keep.rstrip("*")):
                        print("    ", k)
                        _new_dict[k] = v
            elif keep in sample_dict.keys():
                print("    ", keep) 
                _new_dict[keep] = sample_dict[keep]
            else:
                for key in sample_dict.keys():
                    if keep in sample_dict[key]:
                        _new_dict[key] = [keep]
        sample_dict = _new_dict
        print("X", sample_dict.keys())

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
        if len(all_invalid) > 0:
            if ask_user(f"Do you want to remove {len(all_invalid)} bad files?"):
                print("Removing:")
                for fi in all_invalid:
                    print(f"Removing: {fi}")
                    os.system(f'rm {fi}')
        sys.exit(0)


    start = time.time()
    processor_object = HbbProcessor(tagger=args.tagger, year=args.year, systematics=args.systs,
                                    nnlops_rew=args.rew, skipJER=not args.jec, tightMatch=args.tightMatch,
                                    newTrigger=args.newTrigger, looseTau=args.looseTau,
                                    jet_arbitration=args.arb,
                                    newVjetsKfactor=args.newvjets,
                                    ak4tagger=args.tag,
                                    skipRunB=args.skipRunB,
                                    finebins=args.finebins,
                                    ewkHcorr=args.ewkHcorr,
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
                                        'skipbadfiles': args.skip,
                                        'savemetrics': True,
                                        'schema': processor.NanoAODSchema,
                                        'workers': args.workers},
                                    chunksize=args.chunk, maxchunks=args.max,
                                    #shuffle=True,
                                    )
        
        pprint.pprint(metrics)
        save(output, args.output)

    elif args.executor == 'parsl':
        import parsl
        from parsl.app.app import python_app, bash_app
        from parsl.configs.local_threads import config
        from parsl.providers import LocalProvider, CondorProvider, SlurmProvider
        from parsl.channels import LocalChannel
        from parsl.config import Config
        from parsl.executors import HighThroughputExecutor
        from parsl.executors.threads import ThreadPoolExecutor
        from parsl.launchers import SrunLauncher
        from parsl.monitoring.monitoring import MonitoringHub

        from parsl.addresses import address_by_hostname

        x509_proxy = 'x509up_u%s'%(os.getuid())
        wrk_init = '''
        export XRD_RUNFORKHANDLER=1
        export X509_USER_PROXY=/tmp/x509up_u27388
        export X509_CERT_DIR=/home/anovak/certs/
        ulimit -u 32768
        '''
        if args.monitor:
            monitoring = MonitoringHub(
                hub_address=address_by_hostname(),
                hub_port=55055,
                monitoring_debug=False,
                resource_monitoring_interval=5,
            )
        else:
            monitoring = None


        def retry_handler(exception, task_record):
            from parsl.executors.high_throughput.interchange import ManagerLost
            if isinstance(exception, ManagerLost):
                return 0.1
            else:
                return 1

        slurm_htex = Config(
            executors=[
                HighThroughputExecutor(
                    label="jobs",
                    address=address_by_hostname(),
                    prefetch_capacity=0,
                    worker_debug=True,
                    provider=SlurmProvider(
                        channel=LocalChannel(script_dir='logs_parsl'),
                        launcher=SrunLauncher(),
                        max_blocks=args.workers,
                        init_blocks=args.workers,
                        partition='all',
                        # scheduler_options=sched_opts,   # Enter scheduler_options if needed
                        worker_init=wrk_init, 
                        walltime='03:00:00'
                    ),
                ),
                HighThroughputExecutor(
                    label="merges",
                    address=address_by_hostname(),
                    prefetch_capacity=0,
                    worker_debug=True,
                    provider=SlurmProvider(
                        channel=LocalChannel(script_dir='logs_parsl'),
                        launcher=SrunLauncher(),
                        max_blocks=1,
                        init_blocks=1,
                        partition='all',
                        # scheduler_options=sched_opts,   # Enter scheduler_options if needed
                        worker_init=wrk_init, 
                        walltime='03:00:00'
                    ),
                ),
                # HighThroughputExecutor(
                #     label="merges",
                #     worker_debug=True,
                #     cores_per_worker=1,
                #     provider=LocalProvider(
                #         channel=LocalChannel(),
                #         init_blocks=1,
                #         max_blocks=10,
                #     ),
                # ),
            ],
            monitoring=monitoring,
            retries=2,
            retry_handler=retry_handler,
        )
        dfk = parsl.load(slurm_htex)

        if args.chunkify:
            print("XXX")
            # os.mkdir("temp")
            for key, fnames in sample_dict.items():
                output, metrics = processor.run_uproot_job({key: fnames},
                                    treename='Events',
                                    processor_instance=processor_object,
                                    executor=processor.parsl_executor,
                                    executor_args={
                                        'skipbadfiles': args.skip,
                                        'savemetrics':True,
                                        'schema': processor.NanoAODSchema,
                                        # 'mmap':True,
                                        'config': None},
                                    #chunksize=args.chunk, maxchunks=args.max
                                    chunksize=args.chunk, maxchunks=args.max
                                    )

                pprint.pprint(metrics, compact=True)
                save(output, f"temp/{key}.coffea")

                print("X-size", len(pickle.dumps(output))/(1024*1024))

        elif args.merges:
            output, metrics = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=processor_object,
                                    executor=processor.parsl_executor,
                                    executor_args={
                                        'skipbadfiles': args.skip,
                                        'savemetrics': True,
                                        'schema': processor.NanoAODSchema,
                                        'merging': True,
                                        'merges_executors': ['merges'],
                                        'jobs_executors': ['jobs'],
                                        'config': None},
                                    chunksize=args.chunk, maxchunks=args.max
                                    )
        else:
            output, metrics = processor.run_uproot_job(sample_dict,
                                    treename='Events',
                                    processor_instance=processor_object,
                                    executor=processor.parsl_executor,
                                    executor_args={
                                        'skipbadfiles': args.skip,
                                        'savemetrics': True,
                                        'schema': processor.NanoAODSchema,
                                        # 'mmap':True,
                                        'config': None},
                                    chunksize=args.chunk, maxchunks=args.max
                                    )
        pprint.pprint(metrics, compact=True)
        save(output, args.output)

        import datetime
        print("Process time:")
        print(str(datetime.timedelta(seconds=metrics['processtime'])))

    elif args.executor == 'dask':
        from dask_jobqueue import SLURMCluster
        from distributed import Client
        from dask.distributed import performance_report

        cluster = SLURMCluster(
            scheduler_options={
                "dashboard_address": f':{args.dash}',
                'allowed_failures': 50
            },
            queue='all',
            cores=10,
            processes=5,
            memory="500GB",
            retries=10,
            walltime='00:60:00',
            env_extra=['ulimit -u 16000', 'ulimit -n 8000'],
            extra=["--lifetime", "60m", "--lifetime-stagger", "4m"],
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
                                            'skipbadfiles': args.skip,
                                            'schema': processor.NanoAODSchema,
                                            "treereduction": 4,
                                        },
                                        chunksize=args.chunk, maxchunks=args.max
                            )

        save(output, args.output)

    print("Run time:")
    end = time.time()
    print("TIME:", time.strftime("%H:%M:%S", time.gmtime(end-start)))

    with open(f"metrics_{args.output.split('.')[0]}.json", 'w') as metrics_out:
        json.dump(metrics, metrics_out, indent=4)

