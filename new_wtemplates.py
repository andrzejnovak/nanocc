import os
import sys
import time
import collections
import warnings
import copy
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
import re
import json
import hist
import numpy as np
from coffea.util import load, save

import matplotlib.pyplot as plt
import mplhep as hep
from tqdm.auto import tqdm

s = hist.tag.Slicer()

from new_templates import xSecReader, getSumW, scaleSumW, collate, export1d


def make_wtemplates(collated, temptype, lumi=1, systs=True, region='wtag'):
    odict = {}
    proc_names = list(collated.keys())

    for proc in tqdm(proc_names, desc='Samples:'):
        h_obj = copy.deepcopy(collated[proc]['wtag'])
        if systs:
            sys_list = h_obj.axes['systematic']
        else:
            sys_list = ['nominal']
        for syst in tqdm(sys_list, desc='Syst', leave=False):
            base_cuts = {'region': region, 'systematic': syst}
            if temptype == 'cvl':
                # Pass CvL, Pass CvB + N2
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:0j:sum], 'ddc': s[0.45j:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail CvL, Pass CvB + N2
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:0j:sum], 'ddc': s[0:0.45j:sum]}}
                fail_temp = h_obj[cdict]
            elif temptype == 'cvb':
                # Pass CvB
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail CvB 
                cdict = {**base_cuts, **{'ddcvb': s[0:0.03j:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0:len:sum]}}
                fail_temp = h_obj[cdict]
            elif temptype == 'cvbcvl':
                # Pass CvL, Pass CvB
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0.45j:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail CvL, Pass CvB 
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0:0.45j:sum]}}
                fail_temp = h_obj[cdict]
            elif temptype == 'n2cvb':
                # Pass CvB + N2
                cdict = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0:0j:sum], 'ddc': s[0:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail CvB or N2
                cdict_pf = {**base_cuts, **{'ddcvb': s[0.03j:len:sum], 'n2ddt':s[0j:len:sum], 'ddc': s[0:len:sum]}}
                cdict_fp = {**base_cuts, **{'ddcvb': s[0:0.03j:sum], 'n2ddt':s[0:0j:sum], 'ddc': s[0:len:sum]}}
                cdict_ff = {**base_cuts, **{'ddcvb': s[0:0.03j:sum], 'n2ddt':s[0j:len:sum], 'ddc': s[0:len:sum]}}
                fail_temp = h_obj[cdict_pf] + h_obj[cdict_fp] + h_obj[cdict_ff]
            elif temptype == 'n2':
                # Pass N2
                cdict = {**base_cuts, **{'n2ddt':s[0:0j:sum], 'ddcvb': s[0:len:sum], 'ddc': s[0:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail N2
                cdict = {**base_cuts, **{'n2ddt':s[0j:len:sum], 'ddcvb': s[0:len:sum], 'ddc': s[0:len:sum]}}
                fail_temp = h_obj[cdict]
            elif temptype == 'cvlonly':
                # Pass CvB
                cdict = {**base_cuts, **{'ddcvb': s[0:len:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0.45j:len:sum]}}
                pass_temp = h_obj[cdict]
                # Fail CvB
                cdict = {**base_cuts, **{'ddcvb': s[0:len:sum], 'n2ddt':s[0:len:sum], 'ddc': s[0:0.45j:sum]}}
                fail_temp = h_obj[cdict]
            else:
                raise ValueError(f"Template type ``{temptype}`` not understood")
            if proc not in ['data_obs', "JetHT", 'SingleMuon']:
                pass_temp = pass_temp * lumi
                fail_temp = fail_temp * lumi

            pass_name = f"{proc}_pass_{syst}"
            fail_name = f"{proc}_fail_{syst}"

            odict[pass_name] = pass_temp
            odict[fail_name] = fail_temp

    return odict

if __name__ == "__main__":
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

    parser.add_argument('--id', '--identifier', dest='identifier', default=r'', help='File identifier to carry through (default: %(default)s)')
    parser.add_argument("--year", default=2017, type=int, required=True, choices=[2016, 2017, 2018], help="Scale by appropriate lumi")
    parser.add_argument("-m", "--mergemap", dest='mergemap', default=None, required=True, type=str, help="Load a mergemap json")
    parser.add_argument("--plot", type=str2bool, default='True', choices={True, False}, help='Make plots')
    parser.add_argument("--clip", type=str2bool, default='True', choices={True, False}, help='Clip jet mass range.')
    parser.add_argument("--systs", type=str2bool, default='False', choices={True, False}, help='Process systematics')
    parser.add_argument("-j", "--workers", default=0, type=int, help="Parallelize (N/I)")
    parser.add_argument("--region", default='wtag', choices=['wtag', 'wtag2'], type=str, help="Selection region")
    parser.add_argument("--type", default='n2', choices=['n2', 'n2cvb', 'cvb', 'cvl', 'cvbcvl', 'cvlonly'], type=str, help="Template selection type")
    parser.add_argument("-o", "--out", dest='output', default=None, type=str, help="Output file name eg `templates.root`")

    args = parser.parse_args()
    print("Running with the following options:")
    print(args)

    start = time.time()
    lumi = {
        2016: 35.5,
        2017: 41.5,
        2018: 59.2,
    }
    lumi_mu = {
        2016: 35.2,
        2017: 41.1,
        2018: 59.0,
    }

    if args.output is None:
        if not os.path.exists(args.identifier):
            os.mkdir(args.identifier)
        if not os.path.exists(os.path.join(args.identifier, 'plots')):
            os.mkdir(os.path.join(args.identifier, 'plots'))
        template_file = f"{args.identifier}/wtemplates_{args.type}.root"
    else:
        _base_name = args.output.split(".root")[0]
        template_file = f"{_base_name}.root"

    # Load info
    print(f'Processing coffea output from: hists_{args.identifier}.coffea')
    output = load(f'hists_{args.identifier}.coffea')

    xsecs = xSecReader('metadata/xSections_manual.dat')
    sumw = getSumW(output)

    if args.mergemap is not None:
        print(f'Processing with mergemap from: {args.mergemap}')
        with open(args.mergemap) as json_file:
            merge_map = json.load(json_file)
    else:
        merge_map = None

    soutput = scaleSumW(output, sumw, xs=xsecs)
    collated = collate(soutput, merge_map)
    # Rename data
    collated['data_obs'] = collated['SingleMuon']

    # Per sample templates
    templates = make_wtemplates(collated, args.type, lumi=lumi_mu[args.year], systs=args.systs, region=args.region)

    # Stack them to matched/unmatched
    merged_dict = {}
    samples = ['tqq', 'stqq', 'wln', 'zll', 'vvqq']
    for syst in sorted(list(set([k.split('tqq_pass_')[-1] for k in templates.keys() if 'tqq_pass' in k]))):
        for pf in ['pass', 'fail']:
            if syst == 'nominal':
                data = templates[f'data_obs_{pf}_{syst}'][{'genflavor': s[::sum]}]
            matched = sum([templates[f'{sname}_{pf}_{syst}'][{'genflavor': s[1::sum]}] for sname in samples])
            unmatched = sum([templates[f'{sname}_{pf}_{syst}'][{'genflavor': s[:1:sum]}] for sname in samples])

            matched_name = f"catp2_{pf}_{syst}"
            unmatched_name = f"catp1_{pf}_{syst}"
            if syst == 'nominal':
                data_name = f"data_obs_{pf}_{syst}"
                merged_dict[data_name] = data

            merged_dict[matched_name] = matched
            merged_dict[unmatched_name] = unmatched

    print(f'Will save templates to {template_file}')
    fout = uproot3.recreate(template_file)
    for name, h_obj in tqdm(merged_dict.items(), desc='Writing templates'):
        if np.sum(h_obj.values()) <= 0.:
            print(f'Template {name} is empty')
        if args.clip:
            h_obj = h_obj[40j:145j]
        fout[name] = export1d(h_obj)
    fout.close()

    if args.plot:
        hep.style.use("CMS")

        for clip in [True, False]:
            for pf in ["Pass", "Fail"]:
                # Make template plots
                fig, ax = plt.subplots()
                hep.histplot(
                    [merged_dict[f'catp1_{pf.lower()}_nominal'], merged_dict[f'catp2_{pf.lower()}_nominal']],
                    stack=True,
                    label=['Unmatched', 'Matched'])
                hep.histplot(merged_dict[f'data_obs_{pf.lower()}_nominal'],
                             histtype='errorbar', color='k', label='Data',
                             capsize=4, elinewidth=1.2, markersize=12)

                plt.legend(title=pf)
                plt.xlabel(r'jet $m_{SD}$')
                if clip:
                    plt.xlim(40, 145)
                else:
                    plt.xlim(40, 200)
                hep.cms.label('Preliminary', year=args.year, data=True)
                _cl = "_clip" if clip else ""
                fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_templates_{pf.lower()}{_cl}.png")
                fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_templates_{pf.lower()}{_cl}.pdf")

                # Make sample plots
                fig, ax = plt.subplots()
                mc = [templates[f'{sname}_{pf.lower()}_nominal'][{'genflavor': s[::sum]}] for sname in samples[::-1]]
                hep.histplot(mc, stack=True, label=samples[::-1], histtype='fill')
                hep.histplot(templates[f'data_obs_{pf.lower()}_nominal'][{'genflavor': s[::sum]}],
                             histtype='errorbar', color='k', label='Data',
                             capsize=4, elinewidth=1.2, markersize=12)
                plt.legend(title=pf)
                plt.xlabel(r'jet $m_{SD}$')
                if clip:
                    plt.xlim(40, 145)
                else:
                    plt.xlim(40, 200)
                hep.cms.label('Preliminary', year=args.year, data=True)
                _cl = "_clip" if clip else ""
                fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_samples_{pf.lower()}{_cl}.png")
                fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_samples_{pf.lower()}{_cl}.pdf")

                # Make RAW sample plots
                # if True:
                #     soutput['data_obs'] = soutput['SingleMuon']
                #     raw_templates = make_wtemplates(soutput, args.type, lumi=lumi_mu[args.year], systs=args.systs, region=args.region)
                #     fig, ax = plt.subplots()
                #     allsamples = [sname for sname in list(soutput.keys())[::-1] if sname not in ['SingleMuon', 'JetHT']]
                #     mc = [raw_templates[f'{sname}_{pf.lower()}_nominal'][{'genflavor': s[::sum]}] for sname in allsamples]
                #     hep.histplot(mc, stack=True, label=allsamples[::-1], histtype='fill')
                #     hep.histplot(raw_templates[f'data_obs_{pf.lower()}_nominal'][{'genflavor': s[::sum]}],
                #                  histtype='errorbar', color='k', label='Data',
                #                  capsize=4, elinewidth=1.2, markersize=12)
                #     plt.legend(title=pf)
                #     plt.xlabel(r'jet $m_{SD}$')
                #     if clip:
                #         plt.xlim(40, 145)
                #     else:
                #         plt.xlim(40, 200)
                #     hep.cms.label('Preliminary', year=args.year, data=True)
                #     _cl = "_clip" if clip else ""
                #     fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_raw_{pf.lower()}{_cl}.png")
                #     fig.savefig(f"{args.identifier}/plots/wtemplates_{args.type}_raw_{pf.lower()}{_cl}.pdf")

    print("TIME:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
