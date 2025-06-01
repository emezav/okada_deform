#!/usr/bin/env bash

echo $#

if [ $# -eq 0 ];then
  echo "A configuration file must be supplied. See samples/ folder for some examples."
  exit 1
fi

echo "Scenario: $1"

# Parameters: Scenario configuration file path

./okada_deform $1