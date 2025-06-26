#!/bin/bash

# if not there, get BBB file
if [ ! -f "BBB Fossil Transition Analysis Inputs.xlsm" ]; then
    {
      etb cloud get "patio-restricted/BBB Fossil Transition Analysis Inputs.xlsm"
    } || {
         echo "etb cloud init must run before BBB config can be downloaded"
    }
fi
