# deephc
## Environment requirements
Make sure the server are installed Minimap2, Samtools, Anaconda.  
The environment requirement: python3.6, Tensorflow 2.3  

## Correction steps
Run prepare correction_data_pb.sh to correct a pacbio long reads, the same execution as ONT reads  
the script includes the process of evaluation.  

## Overview the files
mp_pileup_hdf_label_SR_ref.py is used to encode the pileup file  
try_tf_model.py is used to train the model  
tf_infer_4.0.py is used to predict and decode the corrected reads  
coverage_count2.0.py is used to count the coverage of the long reads
