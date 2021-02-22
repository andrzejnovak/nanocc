
for year in 16 17 18;
    do 

    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_n2/wtag_pass.root  --plot
    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_n2/wtag_fail.root  --plot

    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_cvl/wtag_pass.root  --plot
    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_cvl/wtag_fail.root  --plot

    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_n2cvb/wtag_pass.root  --plot
    python scalesmear.py -i wfitsjer${year}/wfit_nskim${year}_n2cvb/wtag_fail.root  --plot

done