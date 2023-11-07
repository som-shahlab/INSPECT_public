#!/bin/bash



echo "***************" "./run_classify_pe.sh" "********************************************"
./run_classify_pe.sh &&
pid1=$!

echo "***************" "./run_classify_1m_mort.sh" "********************************************"
./run_classify_1m_mort.sh &&
pid2=$!

echo "***************" "./run_classify_6m_mort.sh" "********************************************"
./run_classify_6m_mort.sh &&
pid3=$!

echo "***************" "./run_classify_12m_mort.sh" "********************************************"
./run_classify_12m_mort.sh &&
pid4=$!

echo "***************" "./run_classify_read_1m.sh" "********************************************"
./run_classify_read_1m.sh &&
pid5=$!

echo "***************" "./run_classify_read_6m.sh" "********************************************"
./run_classify_read_6m.sh &&
pid6=$!

echo "***************" "./run_classify_read_12m.sh" "********************************************"
./run_classify_read_12m.sh &&
pid7=$!

echo "***************" "./run_classify_ph.sh" "********************************************"
./run_classify_ph.sh &&
pid8=$!

