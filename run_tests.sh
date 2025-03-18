#!/bin/bash

# Array of algorithms to test
algorithms=("BFS" "DFS" "UCS" "ASTAR")

# Loop through test cases 0 to 3
for test_number in {0..3}; do
  echo "Running test case $test_number"
  
  # Loop through each algorithm
  for algo in "${algorithms[@]}"; do
    echo "Running $algo on test case $test_number"
    
    # Run the python script for each test and algorithm combination
    python LocalTest.py $test_number $algo
    
    # Add a separator between tests for better readability
    echo "-----------------------------"
  done
done
