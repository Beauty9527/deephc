
# prepare_data.sh include aligned the reference and short reads to the long reads to be corrected.
# then merge the two mpileup, divied the mpileup by linecounts to adapt the multi preprocessing.
# generate a whole hdf for train and a set of split_hdf for infer
# sh prepare_data.sh F.antasticus_long_error.fa F.antasticus_short.fa F.antasticus_genome.fa


long_reads = $1
short_reads = $2
reference = $3

mkdir split_mpileup
mkdir split_hdf
mkdir split_fasta

conda activate tfgpu2.3p3.6
source ~/.bash_profile

# align and generate a complete mpileup
minimap2 -x map-ont -t 30 $1 $3 --MD -a --secondary=no -o truth2draft.bam 
samtools sort truth2draft.bam -o truth2draft.sorted.bam
samtools index truth2draft.sorted.bam
samtools mpileup -a -f $1 truth2draft.sorted.bam -o truth2draft.txt -a

minimap2 -x map-ont -t 30 $1 $2 --MD -a --secondary=no -o subreads2draft.bam 
samtools sort subreads2draft.bam -o subreads2draft.sorted.bam
samtools index subreads2draft.sorted.bam
samtools mpileup -a -f $1 subreads2draft.sorted.bam -o subreads2draft.txt -a

samtools mpileup subreads2draft.sorted.bam truth2draft.sorted.bam -a -s -f $1 -o mpileup_genome.pileup -a

# split the mpileup into a folder for multi-threads
python split_file.py --mpileup mpileup_genome.pileup --output-folder ./split_mpileup
rm -rf ./split_mpileup/manifest

# encode the splited mpileup into a complete HDF file and write the uncovered sequences out uncovered.fasta for train
python mp_pileup_hdf_label_SR_ref.py --mpileup-folder ./split_mpileup --output-file covered_fasta.hdf --uncovered-fasta uncover.fasta

# encode the splited mpileup into splited HDF files for infer
python split_mpileup_encode1.0.py --mpileup-folder ./split_mpileup --output-folder ./split_hdf

# train
python try_tf_model.py --hdf-files covered_fasta.hdf --check-point-path ./weight/cp.ckpt --model_hdf5_path ./weight/model.h5

# infer to correctio 
python tf_infer_4.0.py --hdf-folder ./split_hdf --output-folder ./split_fasta --model-hdf5-path ./weight/model.h5


cat split_fasta/*fasta > corrected_covered.fasta

# merge the uncovered reads to corrected reads and evaluate
 cat uncover.fasta corrected_covered.fasta > corrected.fasta

# evaluate the results

sh evaluate.sh $3 corrected.fasta ont


rm -rf *.bam *.fai *.bai
rm -rf split_fasta







