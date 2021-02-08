import os
import sys
import shutil
import warnings
import uproot
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from uproot3_methods.classes.TH1 import Methods as TH1Methods
    import uproot3
from coffea import hist
from coffea.util import load, save
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

import argparse
parser = argparse.ArgumentParser()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser.add_argument('--id', '--identifier', dest='identifier', required=True, help='File identifier to carry through (default: %(default)s)')
parser.add_argument("--type", default='n2cvb',  choices=['n2', 'n2cvb'], type=str, help="Which region to run")
parser.add_argument('--plot', action='store_true', help='Make control plots')
args = parser.parse_args()
print("Running with the following options:")
print(args)


from coffea.util import load
output = load(f'scaled_hists_{args.identifier}.coffea')

store_dir = f'wfit_{args.identifier}_{args.type}'
if os.path.exists(store_dir):
    shutil.rmtree(store_dir)
os.mkdir(store_dir)

if args.type == 'n2cvb':
    Pass = (output['wtag_opt'] # PP
            .integrate('region', 'wtag')
            .integrate('systematic', 'nominal')
            .rebin('msd', hist.Bin('msd', 'msd', 18, 40, 136.6))
            .integrate('n2ddt', slice(None, 0))
            .integrate('ddcvb', slice(0.017, None))
            [['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu', 'data_obs']]
            .project('msd','genflavor', 'process'))

    Fail = (
        (output['wtag_opt'] # PF
            .integrate('region', 'wtag')
            .integrate('systematic', 'nominal')
            .rebin('msd', hist.Bin('msd', 'msd', 18, 40, 136.6))
            .integrate('n2ddt', slice(None, 0))
            .integrate('ddcvb', slice(None, 0.017))
            [['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu', 'data_obs']]
            .project('msd', 'genflavor', 'process')) 
        + (output['wtag_opt'] # FP/FF
            .integrate('region', 'wtag')
            .integrate('systematic', 'nominal')
            .rebin('msd', hist.Bin('msd', 'msd', 18, 40, 136.6))
            .integrate('n2ddt', slice(0, None))
            [['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu', 'data_obs']]
            .project('msd', 'genflavor', 'process'))
    )
elif args.type == 'n2':
    Pass = (output['wtag_opt'] # P
            .integrate('region', 'wtag')
            .integrate('systematic', 'nominal')
            .rebin('msd', hist.Bin('msd', 'msd', 18, 40, 136.6))
            .integrate('n2ddt', slice(None, 0))
            [['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu', 'data_obs']]
            .project('msd','genflavor', 'process'))

    Fail = (output['wtag_opt'] # F
            .integrate('region', 'wtag')
            .integrate('systematic', 'nominal')
            .rebin('msd', hist.Bin('msd', 'msd', 18, 40, 136.6))
            .integrate('n2ddt', slice(0, None))
            [['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu', 'data_obs']]
            .project('msd', 'genflavor', 'process'))
else:
    raise NotImplementedError

mclist = ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'qcd', 'ST', 'WJetsToLNu']
mclistnoqcd = ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'ST', 'WJetsToLNu']

temp_data_pass = Pass['data_obs'].integrate('genflavor').project('msd')
temp_data_fail = Fail['data_obs'].integrate('genflavor').project('msd')

temp_unmatched_pass = Pass[mclistnoqcd].integrate('genflavor', 0).project('msd')
temp_unmatched_fail = Fail[mclistnoqcd].integrate('genflavor', 0).project('msd')

temp_matched_pass = Pass[mclistnoqcd].integrate('genflavor', slice(1, 4)).project('msd')
temp_matched_fail = Fail[mclistnoqcd].integrate('genflavor', slice(1, 4)).project('msd')

print("Fit templates")
print("Data")
print(temp_data_pass.project('msd').values()[()])
print(temp_data_fail.project('msd').values()[()])
print("Unmatched")
print(temp_unmatched_pass.project('msd').values()[()])
print(temp_unmatched_fail.project('msd').values()[()])
print("Matched")
print(temp_matched_pass.project('msd').values()[()])
print(temp_matched_fail.project('msd').values()[()])

if args.plot:
    # Matched 
    for stack in True, False:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        hist.plot.plot1d(Pass[mclistnoqcd].integrate('genflavor', slice(1, 4)), ax=ax1, stack=stack)
        hist.plot.plot1d(Fail[mclistnoqcd].integrate('genflavor', slice(1, 4)), ax=ax2, stack=stack)
        fig.savefig(f'{store_dir}/matched{stack}.png')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        hist.plot.plot1d(Pass[mclistnoqcd].integrate('genflavor', 0), ax=ax1, stack=stack)
        hist.plot.plot1d(Fail[mclistnoqcd].integrate('genflavor', 0), ax=ax2, stack=stack)
        fig.savefig(f'{store_dir}/unmatched{stack}.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    hist.plot.plot1d(Pass['data_obs'].integrate('genflavor', 0), ax=ax1, error_opts={'color':'black'})
    hist.plot.plot1d(Fail['data_obs'].integrate('genflavor', 0), ax=ax2, error_opts={'color':'black'})
    hist.plot.plot1d(Pass[mclist].integrate('genflavor'), ax=ax1, clear=False, stack=True)
    hist.plot.plot1d(Fail[mclist].integrate('genflavor'), ax=ax2, clear=False, stack=True)
    fig.savefig(f'{store_dir}/datamc.png')

    hep.set_style('CMS')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    hep.histplot(temp_data_pass.values()[()], 
                 temp_data_pass.axis('msd').edges(),
                 ax=ax1, histtype='errorbar', color='black', label='Data', yerr=True)
    hep.histplot([temp_unmatched_pass.values()[()], temp_matched_pass.values()[()]], 
                 temp_matched_pass.axis('msd').edges(),
                 stack=True, ax=ax1, label=['Unmatched', 'Matched'], histtype='fill', color=['green', 'red'])
    hep.histplot(temp_data_fail.values()[()], 
                 temp_data_fail.axis('msd').edges(),
                 ax=ax2, histtype='errorbar', color='black', label='Data', yerr=True)
    hep.histplot([temp_unmatched_fail.values()[()], temp_matched_fail.values()[()]], 
                 temp_matched_fail.axis('msd').edges(),
                 stack=True, ax=ax2, label=['Unmatched', 'Matched'], histtype='fill', color=['green', 'red'])

    for ax in ax1, ax2:
        ax.legend()
        ax.set_xlabel('jet $m_{SD}$')
    ax1.set_title("pass")
    ax2.set_title("fail")
    fig.savefig(f'{store_dir}/templates.png')

fout_pass = uproot3.create(f'{store_dir}/wtag_pass.root')
fout_fail = uproot3.create(f'{store_dir}/wtag_fail.root')

fout_pass['data_obs'] = hist.export1d(temp_data_pass)
fout_fail['data_obs'] = hist.export1d(temp_data_fail)
fout_pass['catp2'] = hist.export1d(temp_matched_pass)
fout_fail['catp2'] = hist.export1d(temp_matched_fail)
fout_pass['catp1'] = hist.export1d(temp_unmatched_pass)
fout_fail['catp1'] = hist.export1d(temp_unmatched_fail)

fout_pass.close()
fout_fail.close()