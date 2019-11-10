# FAQ
1. Do you have other download sources?
    <!-- * Our dataset can be downloaded in HKUST OneDrive [here](https://gohkust-my.sharepoint.com/:f:/g/personal/saikit_ust_hk/EqRFLP5XEihCt_PFIHyPNO8BsKb7r8S5V5ELaCqk7UdDTQ?e=FX2OPF) or from the SUTD server [here](http://103.24.77.34:8080/scanobjectnn/).-->
    * We are currently tidying up the data for public release, if you need an early access for academic purposes, please send an email to mikacuy@gmail.com.
2. How to evaluate on scanobjectnn with model trained on ModelNet40?
    * Please see the file "evaluate_real_trained_on_synthetic.py". You can see the class mapping file in "mapping2.py".
3. What is the difference between main_split,split1,split2,split3 and split4?
    * "Main_split" was used for the experiments in our main paper, while the other splits (1-4) are additional splits that we reported in our supplementary materials.
4. H5 labels: Can you confirm me that the label-integer correspondence (in h5 splits) is the one from 'shape_names_ext.txt' in ascending order and starting from zero?
e.g.
{bag: 0, bin: 1, box: 2, cabinet: 3, chair: 4, desk: 5, display: 6, door: 7, shelf: 8, table: 9, bed:	10, pillow: 11, sink:	12, sofa:	13, toilet:	14}
    * Yes, that is correct.
5. I want to assure that you use -1 to denote background in the mask.
    * Yes, that's right.
