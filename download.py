import os
import json
import argparse
import pprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run analysis on baconbits files using processor coffea files')
    parser.add_argument('-i', '--input', default=r'metadata/dataset.json', help='')
    parser.add_argument('-d', '--dir', help='Storage directory', required=True)
    parser.add_argument('-o', '--output', default=r'metadata/dataset_local.json', help='')
    parser.add_argument('--download', help='Bool', action='store_true')
    
    args = parser.parse_args()
    
    # load dataset
    with open(args.input) as f:
        sample_dict = json.load(f)

    print("Storage dir:")
    print("   ", os.path.abspath(args.dir))

    out_dict = {}
    for key in sample_dict.keys():
        new_list = []
        print(key)
        for fname in sample_dict[key]:
            out = os.path.join(os.path.abspath(args.dir), fname.split("//")[-1].lstrip("/"))
            new_list.append(out)
            if args.download:
                if os.path.isfile(out):
                    'File found'
                else:
                    #print("xrdcp -P " + fname + " " + out)
                    os.system("xrdcp -P " + fname + " " + out)
        out_dict[key] = new_list

    print("Writing files to {}".format(args.output))
    with open(args.output, 'w') as fp:
        json.dump(out_dict, fp, indent=4)
        