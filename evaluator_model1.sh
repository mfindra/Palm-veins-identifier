#!/bin/bash

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' 

total_matches=0
total_nonmatches=0

for identifier in $(seq -w 1 100); do
    for element in 04 05 06; do
        jpg_path="./dataset/images/${identifier}_l_940_${element}.jpg"
        command="./.venv/bin/python ./script_model1.py --match \"./dataset/images_l_940_train_marked/\" \"$jpg_path\""
        
        output=$(eval $command)
        rank1=$(echo "$output" | grep "Rank 1:" | awk '{print $4}')
        group=$(echo "$output" | grep "Rank 1:" | awk '{print $6}')
        

        if [ $rank1 == "${identifier}_l_940" ]; then
            echo -e "${GREEN}Match!${NC} Rank 1: $rank1, Image: $identifier, Match: $group"
            ((total_matches++))
        else
            echo -e "${RED}No match.${NC} Rank 1: $rank1, Image: $identifier, Match: $group"
            ((total_nonmatches++))
        fi
    done
done

total_images=$((total_matches + total_nonmatches))
percentage=$((total_matches * 100 / total_images))

echo -e "\n\nMODEL 1"
echo -e "\nTotal Matches: $total_matches"
echo -e "Total Non-Matches: $total_nonmatches"
echo -e "Percentage of Matches: $percentage%"