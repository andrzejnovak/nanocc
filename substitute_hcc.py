import os
import argparse
import uproot
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import uproot3
from coffea import hist
parser = argparse.ArgumentParser()

lumi = {
    2016 : 35.5,
    2017 : 41.5,
    2018 : 59.2,
}

parser.add_argument("-s", '--source', type=str, required=True, help="Source file")
parser.add_argument("-t", '--target', type=str, required=True, help="Target file")
parser.add_argument("-n", '--new', type=str, default=None, help="Name for modified file")
parser.add_argument("--ys", '--yearsource', default=2017, type=int, help="Scale by appropriate lumi")
parser.add_argument("--yt", '--yeartarget', default=2018, type=int, help="Scale by appropriate lumi")

args = parser.parse_args()
if args.new is None:
    args.new = args.target.rstrip(".root")+"_mod.root"
print("Running with the following options:")
print(args)

fins = uproot3.open(args.source)
fint = uproot3.open(args.target)
if os.path.exists(args.new):
    os.remove(args.new)
fout = uproot3.create(args.new)

for key in fint.keys():
    if b'hcc' in key:
        src_hist = fins[key]
        for i in range(len(src_hist)):
            src_hist[i] /= lumi[args.ys]
            src_hist[i] *= lumi[args.yt]
        fout[key] = src_hist 
    else:
        fout[key] = fint[key]



