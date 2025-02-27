#!/usr/bin/env bash

# delete previous output from PBS
rm -rf *.qsub_out

# submit the job to the queue
qsub submission.pbs

# wait for the job to get picked up and start producing output
until [ -f *.qsub_out ]
do
  sleep 1
done

# open the output file and follow the file as new output is added
less +F *.qsub_out
