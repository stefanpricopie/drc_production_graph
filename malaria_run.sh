#!/bin/bash
set -e
set -o pipefail

# This function launches one job $1 is the job name, the other arguments is the job to submit.
qsub_job() {
   JOBNAME='Run'-$$
   qsub -v PATH <<EOF
#!/bin/bash --login
#$ -t 1-$nruns
#$ -N $JOBNAME
# -l ivybridge
# -l short
# -l mem512
#$ -M stefan.pricopie@postgrad.manchester.ac.uk
# -m se
#      b     Mail is sent at the beginning of the job.
#      e     Mail is sent at the end of the job.
#      a     Mail is sent when the job is aborted or rescheduled.
#      s     Mail is sent when the job is suspended.
#
#$ -o $OUTDIR/${JOBNAME}.txt
#$ -j y
#$ -cwd
run=\$SGE_TASK_ID
echo "running: python malaria.py $@ --seed \$SGE_TASK_ID --output $OUTDIR"
python malaria.py $@ --seed \$SGE_TASK_ID --output $OUTDIR
EOF
}

nruns=40

for file in master_16190_graph_0; do
  for algo in rs lc5 lc20 bo bopu; do
    for b_exp in 1; do
      for c_base in 3; do
        for c_synth in 5; do
          if [[ $OSTYPE == 'linux'* ]]
          then
            OUTDIR="$HOME/scratch/GECCO-extension/output/malaria_new"
            # echo "$file $algo --b_exp $b_exp --c_base $c_base --c_synth $c_synth"
            qsub_job "$file $algo --b_exp $b_exp --c_base $c_base --c_synth $c_synth"
          else
            # TODO: fix local run
            OUTDIR="./output/malaria_corrected"
            echo "running: python malaria.py $file $algo --b_exp $b_exp --c_base $c_base --c_synth $c_synth"
            python malaria_old.py $file $algo --b_exp $b_exp --c_base $c_base --c_synth $c_synth --output $OUTDIR &
          fi
        done
      done
    done
  done
done

if [[ $OSTYPE != 'linux'* ]]
then
  wait
  echo "All done"
fi
