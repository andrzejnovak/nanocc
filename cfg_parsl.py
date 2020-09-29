import parsl
import os
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config

from parsl.providers import LocalProvider,CondorProvider,SlurmProvider
from parsl.channels import LocalChannel,SSHChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher

from parsl.addresses import address_by_hostname

x509_proxy = 'x509up_u%s'%(os.getuid())

#export X509_USER_PROXY=$WORK/%s
wrk_init = '''
export XRD_RUNFORKHANDLER=1
export X509_USER_PROXY=x509up_u31233
export X509_CERT_DIR=/home/anovak/certs/
ulimit -u 32768
'''#%(x509_proxy)

twoGB = 2048
nproc = 36

sched_opts = '''
#SBATCH --cpus-per-task=%d
#SBATCH --mem-per-cpu=%d
''' % (nproc, twoGB, )


slurm_htex = Config(
    executors=[
        HighThroughputExecutor(
            label="coffea_parsl_slurm",
            address=address_by_hostname(),
            prefetch_capacity=0,
            max_workers=nproc,
            #suppress_failure=True,
            provider=SlurmProvider(
                channel=LocalChannel(script_dir='test_parsl'),
                launcher=SrunLauncher(),
                #min_blocks=4,
                #init_blocks=6,
                max_blocks=20,
                init_blocks=16, 
                #nodes_per_block=1,
                partition='all',
                scheduler_options=sched_opts,   # Enter scheduler_options if needed
                worker_init=wrk_init,         # Enter worker_init if needed
                walltime='00:120:00'
            ),
        )
    ],
    retries=20,
    strategy=None,
)

#parsl.set_stream_logger() # <-- log everything to stdout

dfk = parsl.load(slurm_htex)

chunksize=100000
