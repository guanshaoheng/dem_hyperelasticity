#!/bin/bash

echo off

clear

export PYTHONPATH=/home/shguan/dem_hyperelasticity:$PYTHONPATH

python3 main_dem.py
