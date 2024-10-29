#!/bin/bash

PARAM_FILE="params.txt"
RESULT_DIR="results"
SHADER_DIR="kernels"
PARAMS_DIR="params"
TARGET_DIR="target"

function make_even() {
    if (( $1 % 2 == 0 )); then
        echo "$1"
    else
	echo "$(($1 + 1))"
    fi
}

# Generate a random number between min and max
function random_between() {
  local min=$1
  local max=$2

  local range=$((max - min + 1))
  local random=$((RANDOM % range + min))
  echo "$random"
}

function random_config() {
  local workgroupLimiter=$1
  local workgroupSizeLimiter=$2

  echo "testIterations=200" > $PARAM_FILE
  local testingWorkgroups=$(random_between 2 $workgroupLimiter)
  echo "testingWorkgroups=$testingWorkgroups" >> $PARAM_FILE
  local maxWorkgroups=$(random_between $testingWorkgroups $workgroupLimiter)
  echo "maxWorkgroups=$maxWorkgroups" >> $PARAM_FILE
  # ensures total threads is divisible by 2
  local workgroupSize=$(make_even $(random_between 1 $workgroupSizeLimiter))
  echo "workgroupSize=$workgroupSize" >> $PARAM_FILE
  echo "shufflePct=$(random_between 0 100)" >> $PARAM_FILE
  echo "barrierPct=$(random_between 0 100)" >> $PARAM_FILE
  local stressLineSize=$(echo "$(random_between 2 10)^2" | bc)
  echo "stressLineSize=$stressLineSize" >> $PARAM_FILE
  local stressTargetLines=$(random_between 1 16)
  echo "stressTargetLines=$stressTargetLines" >> $PARAM_FILE
  echo "scratchMemorySize=$((32 * $stressLineSize * $stressTargetLines))" >> $PARAM_FILE
  echo "memStride=$(random_between 1 7)" >> $PARAM_FILE
  echo "memStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "memStressIterations=$(random_between 0 1024)" >> $PARAM_FILE
  echo "memStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "preStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "preStressIterations=$(random_between 0 128)" >> $PARAM_FILE
  echo "preStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "stressAssignmentStrategy=$(random_between 0 1)" >> $PARAM_FILE
  echo "permuteThread=419" >> $PARAM_FILE
}

function run_test() {
  local test=$1
  local tb=$2
  local scope=$3
  local fence_scope=$4
  local variant=$5
  local params=$6
  res=$(./$TARGET_DIR/$test-$tb-$scope-$fence_scope-$variant-runner -s $PARAM_FILE -t $PARAMS_DIR/$params)
  local weak_behaviors=$(echo "$res" | tail -n 1 | sed 's/.*of weak behaviors: \(.*\)$/\1/')
  local total_behaviors=$(echo "$res" | tail -n 2 | head -n 1 | sed 's/.*Total behaviors: \(.*\)$/\1/')
  local weak_rate=$(echo "$res" | tail -n 3 | head -n 1 | sed 's/.*rate: \(.*\) per second/\1/')

  echo "  Test $test-$tb-$scope-$fence_scope-$variant weak: $weak_behaviors, total: $total_behaviors, rate: $weak_rate per second"

  if (( $(echo "$weak_rate > 0" | bc -l) )); then
    local test_result_dir="$RESULT_DIR/$test-$tb-$scope-$fence_scope-$variant"
    if [ ! -d "$test_result_dir" ] ; then
      mkdir "$test_result_dir"
      cp $PARAM_FILE "$test_result_dir"
      echo $weak_rate > "$test_result_dir/rate"
    else
      local max_rate=$(cat "$test_result_dir/rate")
      if (( $(echo "$weak_rate > $max_rate" | bc -l) )); then
        cp $PARAM_FILE "$test_result_dir"
        echo $weak_rate > "$test_result_dir/rate"
      fi
    fi
  fi

}

if [ $# != 1 ] ; then
  echo "Need to pass file with test combinations"
  exit 1
fi

if [ ! -d "$RESULT_DIR" ] ; then
  mkdir $RESULT_DIR
fi

if [ ! -d "$TARGET_DIR" ] ; then
  mkdir $TARGET_DIR
fi

tuning_file=$1
readarray test_files < $tuning_file

for test_file in "${test_files[@]}"; do
  read -a test_info <<< "$(sed -n '1p' $test_file)"
  read -a threadblocks <<< "$(sed -n '2p' $test_file)"
  read -a scopes <<< "$(sed -n '3p' $test_file)"
  read -a variants <<< "$(sed -n '4p' $test_file)"

  if [ "$(wc -l < "$test_file")" -ge 5 ]; then
    # Perform the command if the line count condition is met
    read -a fence_scopes <<< "$(sed -n '5p' "$test_file")"
    read -a fence_variants <<< "$(sed -n '6p' $test_file)"
  fi

  test="${test_info[0]}"

  # build binaries
  for tb in ${threadblocks[@]}; do
    for scope in ${scopes[@]}; do
      for variant in ${variants[@]}; do
        echo "Compiling $test-$tb-$scope-NO_FENCE-$variant runner"
  	    nvcc -D$tb -D$scope -D$variant -I. -rdc=true -arch sm_80 runner.cu functions.cu "kernels/$test.cu" -o "$TARGET_DIR/$test-$tb-$scope-NO_FENCE-$variant-runner"
      done
      if [ "$(wc -l < "$test_file")" -ge 5 ]; then
        for f_scope in ${fence_scopes[@]}; do
          for f_variant in ${fence_variants[@]}; do
            echo "Compiling $test-$tb-$scope-$f_scope-$f_variant runner"
  	        nvcc -D$tb -D$scope -D$f_scope -D$f_variant -I. -rdc=true -arch sm_80 runner.cu functions.cu "kernels/$test.cu" -o "$TARGET_DIR/$test-$tb-$scope-$f_scope-$f_variant-runner"
          done
        done
      fi
    done
  done
done

iter=0

while [ true ]
do
  echo "Iteration: $iter"
  random_config 1024 256
  for test_file in "${test_files[@]}"; do
    read -a test_info <<< "$(sed -n '1p' $test_file)"
    read -a threadblocks <<< "$(sed -n '2p' $test_file)"
    read -a scopes <<< "$(sed -n '3p' $test_file)"
    read -a variants <<< "$(sed -n '4p' $test_file)"
    if [ "$(wc -l < "$test_file")" -ge 5 ]; then
      read -a fence_scopes <<< "$(sed -n '5p' "$test_file")"
      read -a fence_variants <<< "$(sed -n '6p' $test_file)"
    fi

    test="${test_info[0]}"
    params="${test_info[1]}"

    for tb in ${threadblocks[@]}; do
      for scope in ${scopes[@]}; do
        for variant in ${variants[@]}; do
          run_test $test $tb $scope NO_FENCE $variant $params
        done
        if [ "$(wc -l < "$test_file")" -ge 5 ]; then
          for f_scope in ${fence_scopes[@]}; do
            for f_variant in ${fence_variants[@]}; do
              run_test $test $tb $scope $f_scope $f_variant $params
            done
          done
        fi
      done
    done
  done
  iter=$((iter + 1))
done

