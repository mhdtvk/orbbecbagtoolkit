#!/bin/bash

# Welcome message
echo -e "\n====== ir_depth_extractor Project ========="

# Get the directory of the Bash script
script_dir="$(dirname "$0")"

# The directory of the Camera Bag files
folder_name="/media/mt/F/BAG files/aug-8-2024/calib_6/"

# Check if the directory exists
if [ ! -d "$folder_name" ]; then
  echo "The directory '$folder_name' does not exist."
  
  # Prompt the user for a new directory
  read -p "Please enter a valid directory of the folder that contains the CAMERAS: " folder_name
fi

# Display default parameters
echo -e "\n[ Default Parameters ]:"
echo -e " < master_camera > = camera_0"
echo -e " < convert_depth_to_8bit > = False"
echo -e " < convert_ir_to_8bit > = False"
echo -e " < policy > = 'drop'"
echo -e " < Batch Size > = 50"

# Prompt the user to change default values
echo -e "\nDo you want to change the default values? Press 'Enter' to use the default values or '1' to change them."
read -p "" user_input

# Check the user's input
if [ "$user_input" == "1" ]; then
  # Prompt the user for new parameters
  read -p "Please enter < master_camera > name: " master_camera
  read -p "Please enter the choice for < convert_depth_to_8bit > (True/False): " convert_depth_to_8bit
  read -p "Please enter the choice for < convert_ir_to_8bit > (True/False): " convert_ir_to_8bit
  read -p "Please enter the choice for < policy >: " policy
  read -p "Please enter the Batch size: " batch_size

  # Run the Python script with user-defined parameters
  python3 "$script_dir/arg_parser.py" -f "$folder_name" -m "$master_camera" -cd "$convert_depth_to_8bit" -ci "$convert_ir_to_8bit" -p "$policy" -b "$batch_size"
else
  # Run the Python script with the default parameters
  python3 "$script_dir/arg_parser.py" -f "$folder_name"
fi

# Wait for user input before exiting
read -p "Press any key to continue..." wait
