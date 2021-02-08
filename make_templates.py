import os
import sys
import collections
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
import re
from coffea import hist
from coffea.util import load, save
import matplotlib.pyplot as plt
import mplhep as hep
from histoprint import print_hist

from runner import show_hists

def print1d(h1d, title=""):
    # try:
    vals = h1d.project('msd').values()[()]
    edges = h1d.project('msd').axis("msd").edges()
    print_hist((vals, edges), title=title, 
            columns=100,
    )
    # except:
    #     print("Couldn't plot", h1d)
     #       bg_colors="rgbcmy")

lumi = {
    2016 : 35.5,
    2017 : 41.5,
    2018 : 59.2,
}
lumi_mu = {
    2016 : 35.2,
    2017 : 41.1,
    2018 : 59.0,
}


def xSecReader(fname):
   # Probably unsafe
    with open(fname) as file:
        lines = file.readlines()
    lines = [l.strip("\n") for l in lines if not l.startswith("#")]    
    lines = [l.split("#")[0] for l in lines if len(l) > 5]
    
    _dict = {}
    for line in lines:
        key = line.split()[0]
        valuex = line.split()[1:]
        if len(valuex) > 1:
            value = "".join(valuex)
        else:
            value = valuex[0]
        _dict[key] = float(eval(value))
    return _dict
    

def rescale(accumulator, xsec):
    """Scale by lumi"""
    lumi = 1000 
    from coffea import hist
    scale = {}
    print("Scaling:")
    for dataset, dataset_sumw in collections.OrderedDict(sorted(accumulator['sumw'].items())).items():
        dataset_key = dataset.lstrip("/").split("/")[0]
        if dataset_key in xsecs:
            print(" ", dataset_key)
            scale[dataset] = lumi*xsecs[dataset_key]/dataset_sumw
        else:
            print(" ", "X ", dataset_key)
            scale[dataset] = 0#lumi / dataset_sumw

    for h in accumulator.values():
        if isinstance(h, hist.Hist):
            h.scale(scale, axis="dataset")
    return accumulator


def collate(hist_obj, auto=True):
    """Merge datsets"""
    name_map = {
        'JetHT': 'data_obs',
        'SingleMuon': 'data_obs',
        'QCD' : "qcd", 
        'ZJetsToQQ': "zqq", 
        'WJetsToQQ': "wqq", 
        'GluGluHToBB': 'hbb',
        'GluGluHToCC': 'hcc',
        'TTToHadronic': 'tqq',
        'TTToSemileptonic': 'tqq',
        'TTTo2L2Nu': 'tqq',
        'ST' : 'stqq',
        'WW': 'vvqq',
        'WZ': 'vvqq',
        'ZZ': 'vvqq',
        'WJetsToLNu' : 'wln',
        'DYJetsToLL' : 'zll',
        }

    import json
    sample_keys = [id.name for id in hist_obj.axis('dataset').identifiers()]

    if auto:
        # Sort and group samples
        from collections import OrderedDict
        import string
        # Get short datset names and create a 1:1 mapping
        short = [k.lstrip('/').split("/")[0].split("_")[0].lstrip('/') for k in sample_keys]
        # Because shit had different names
        for i in range(len(short)):
            if short[i] in name_map.keys():
                short[i] = name_map[short[i]]
        mapper = dict(zip(sample_keys, short))
        # Sort keys and map shortname:list_of_sets
        alphabet = "QTZWG" + string.ascii_uppercase + string.ascii_lowercase
        groups = list(set(mapper.values()))
        map_dict = OrderedDict()
        #for group in sorted(groups, key=lambda word: [alphabet.index(c) for c in word]):
        for group in groups:
            map_dict[group] = [k for k, v in mapper.items() if v == group]

        # Group
        _cat = hist.Cat("process", "Process", sorting='placement')
        hist_obj = hist_obj.group('dataset', _cat, map_dict)
    else:
        raise NotImplementedError

    return hist_obj

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
    parser.add_argument("--year", default=2017, type=int, required=True, help="Scale by appropriate lumi")
    parser.add_argument("--split", type=str2bool, default='True', choices={True, False}, help='Split W/Z by flavour')
    parser.add_argument("--muon", type=str2bool, default='False', choices={True, False}, help='Process muon templates')
    parser.add_argument("--systs", type=str2bool, default='False', choices={True, False}, help='Process systematics')
    parser.add_argument("--type", default='cc',  choices=['cc', 'bb', '3'], type=str, help="B templates or C tempaltes")
    args = parser.parse_args()
    print("Running with the following options:")
    print(args)

    # Bookkeeping
    if args.type == 'cc': 
        file_kind = 'CC'
    elif args.type == '3': 
        file_kind = "3"
    else:
        file_kind = "BB"
    template_file = f"templates_{args.identifier}_{file_kind}.root"
    template_mu_file = f"templatesmuCR_{args.identifier}_{file_kind}.root"
    if os.path.exists(template_file):
        os.remove(template_file)
    if os.path.exists(template_mu_file):
        os.remove(template_mu_file)
    print(f'Will save templates to {template_file}')
    fout = uproot3.create(template_file)

    # Load info
    print(f'Processing coffea output from hists_{args.identifier}.coffea')
    output = load(f'hists_{args.identifier}.coffea')
     
    #xsecs = xSecReader('metadata/xSections.dat')
    xsecs = xSecReader('metadata/xSections_manual.dat')
   
    # Scale by xsecs
    output = rescale(output, xsecs)
    #########
    # Do Signal region
    h = collate(output['templates']).integrate('region', 'signal')
    
    # Scale MC by lumi
    _nodata = re.compile("(?!data_obs)")
    h.scale({p: lumi[args.year] for p in h[_nodata].identifiers('process')}, axis="process")
    
    # Saved scaled hists for debugging
    import copy
    chists = copy.deepcopy(output)
    for key, val in chists.items():
        if isinstance(val, hist.Hist):
            val = collate(val)#.integrate('region', 'signal')
            val.scale({p: lumi[args.year] for p in val[_nodata].identifiers('process')}, axis="process")
            chists[key] = val
    debug_save_name = f'scaled_hists_{args.identifier}.coffea'
    print(f'Saving scaled hists/outputs to {debug_save_name}')
    save(chists, f'{debug_save_name}')

    proc_names = h.axis('process').identifiers()
    from coffea.hist import StringBin
    _extra_procs = {_s: StringBin(_s) for _s in ['wcq', 'zbb', 'zcc']}
    if args.split: proc_names += list(_extra_procs.values())

    totn_proc = len(proc_names) + len(h.identifiers('pt'))
    for proc in proc_names:
        print(proc)
        for i, ptbin in enumerate(h.identifiers('pt')):
            for syst in h.identifiers('systematic'):
                if syst.name != "nominal" and proc == 'data_obs': continue
                source_proc = proc
                if args.split:
                    if proc.name == 'zbb':
                        mproj = (3,)
                    elif proc.name == 'wcq' or proc.name == 'zcc':
                        mproj = (2,)
                    elif proc.name == 'wqq' or proc.name == 'zqq':
                        mproj = (1,)
                    else:
                        mproj = (slice(None), 'all')
                    if proc.name in ['wcq', 'wqq']: source_proc = 'wqq'
                    if proc.name in ['zbb', 'zcc', 'zqq']: source_proc = 'zqq'
                else:
                    mproj = (slice(None), 'all')
                systreal = syst
                if args.type == '3':
                    pqq_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    #.integrate('pxx', 1)
                                    .integrate('ddb', overflow='all')
                                    )
                    pcc_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    #.integrate('pxx', 2)
                                    .integrate('ddb', overflow='all')
                                    )
                    pbb_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    #.integrate('pxx', 3)
                                    .integrate('ddb', overflow='all')
                                    )
                elif args.type == 'cc':
                    fail_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    .integrate('ddb')
                                    #.integrate('ddcvb', slice(0.2, None))
                                    .integrate('ddcvb', slice(0.017, None))
                                    #.integrate('ddcvb')
                                    #.integrate('pxx')
                                    # .integrate('ddc', slice(None,0.83))
                                    .integrate('ddc', slice(None, 0.44))
                                    )
                    pass_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    .integrate('ddb')
                                    # .integrate('ddcvb', slice(0.2, None))   
                                    .integrate('ddcvb', slice(0.017, None))   
                                    #.integrate('ddcvb')
                                    #.integrate('pxx')
                                    # .integrate('ddc', slice(0.83,None))
                                    .integrate('ddc', slice(0.44,None))
                                    )

                else:
                    fail_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    .integrate('ddc')
                                    .integrate('ddcvb')
                                    #.integrate('pxx')
                                    .integrate('ddb', slice(None,0.89))
                                    )
                    pass_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt', ptbin)
                                    .integrate('ddc')
                                    .integrate('ddcvb')   
                                    #.integrate('pxx')
                                    .integrate('ddb', slice(0.89,None))
                                    )
               
                try:
                    content = fail_template.sum('msd').values()
                except:
                    content = pqq_template.sum('msd').values()

                if content == {} or content[()] == 0. and i == 0:
                    print("Missing", proc, ptbin, syst)
                    continue

                sname = "_%s" % syst if syst.name != '' else ''
                if args.type == '3':
                    name = "%s_pqq%s_bin%d" % (proc, sname, i)
                    fout[name] = hist.export1d(pqq_template)
                    name = "%s_pcc%s_bin%d" % (proc, sname, i)
                    fout[name] = hist.export1d(pcc_template)
                    name = "%s_pbb%s_bin%d" % (proc, sname, i)
                    fout[name] = hist.export1d(pbb_template)
                else:
                    name = "%s_pass%s_bin%d" % (proc, sname, i)
                    fout[name] = hist.export1d(pass_template)
                    name = "%s_fail%s_bin%d" % (proc, sname, i)
                    fout[name] = hist.export1d(fail_template)


    fout.close()

    if not args.muon:
        sys.exit()
    ## Muon CR
    print("Muon CR")
    print(f'Will save templates to {template_mu_file}')
    foutmu = uproot3.create(template_mu_file)
    h = collate(output['templates']).integrate('region', 'muoncontrol')

    # Scale MC by lumi
    _nodata = re.compile("(?!data_obs)")
    h.scale({p: lumi_mu[args.year] for p in h[_nodata].identifiers('process')}, axis="process")

    rename = {
        'trigweight': 'trigger',
        'pileupweight': 'Pu',
        'mutrigweight': 'mutrigger',
        'muidweight': 'muid',
        'muisoweight': 'muiso',
        'matchedUp': 'matched',
        'matchedDown': 'unmatched',
    }

    for proc in h.identifiers('process'):
        for syst in h.identifiers('systematic'):
            mproj = (slice(None), 'all')
            systreal = syst
            if args.type == 'cc':
                fail_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt')
                                    .integrate('ddb')
                                    #.integrate('ddcvb', slice(0.2, None))
                                    .integrate('ddcvb', slice(0.017, None))
                                    #.integrate('ddcvb')
                                    #.integrate('pxx')
                                    # .integrate('ddc', slice(None,0.83))
                                    .integrate('ddc', slice(None, 0.44))
                                    #.integrate('ddc')
                                    )
                pass_template = (h.integrate('process', source_proc)
                                    .integrate('genflavor', *mproj)
                                    .integrate('systematic', systreal)
                                    .integrate('pt')
                                    .integrate('ddb')
                                    # .integrate('ddcvb', slice(0.2, None))   
                                    .integrate('ddcvb', slice(0.017, None))   
                                    #.integrate('ddcvb')
                                    #.integrate('pxx')
                                    # .integrate('ddc', slice(0.83,None))
                                    .integrate('ddc', slice(0.44,None))
                                    #.integrate('ddc')
                                    )
            else:
                raise NotImplementedError
                # fail_template = (h.integrate('process', proc)
                #                     .integrate('AK8Puppijet0_isHadronicV', *mproj)
                #                     .integrate('systematic', systreal)
                #                     .integrate('AK8Puppijet0_pt', overflow='all')
                #                     .integrate('pxx')
                #                     .integrate('AK8Puppijet0_deepdoubleb', slice(None,0.89), overflow='under')
                #                 )
                # pass_template = (h.integrate('process', proc)
                #                     .integrate('AK8Puppijet0_isHadronicV', *mproj)
                #                     .integrate('systematic', systreal)
                #                     .integrate('AK8Puppijet0_pt', overflow='all')
                #                     .integrate('pxx')
                #                     .integrate('AK8Puppijet0_deepdoubleb', slice(0.89,None))
                                # )
            content = fail_template.sum('msd').values()
            if content == {} or content[()] == 0.:
                print(proc, syst)
                continue
            sname = "_%s" % syst if syst != '' else ''
            for k,v in rename.items():
                sname = sname.replace(k, v)
            name = "%s_pass%s" % (proc, sname)
            foutmu[name] = hist.export1d(pass_template)
            name = "%s_fail%s" % (proc, sname)
            foutmu[name] = hist.export1d(fail_template)

    foutmu.close()