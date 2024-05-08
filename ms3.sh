#!/bin/bash
clear
# Function to handle keyboard interrupt (Ctrl+C)
interrupt_handler() {
    echo "Keyboard interrupt received. Deleting $OUTPUT_DIR..."
    rm -rf ${OUTPUT_DIR}
    echo "Folder $OUTPUT_DIR deleted."
    cd /home/roberto/PythonProjects/S2RAWVessel
}
# This script installs the MMDET package and its dependencies
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

source $(conda info --base)/etc/profile.d/conda.sh
if conda activate openmmlab; then
    echo "openmmlab environment activated"
else
    echo "openmmlab environment not found"
    exit 1
fi

# Check if the script has been sourced
if [ -z "$BASH_SOURCE" ]; then
    echo "ERROR: You must source this script. Run 'source $0'"
    exit 1
fi
# Check if the script is being run from the right directory
if [ ! -f setup.py ]; then
  echo "ERROR: Run this from the top-level directory of the repository"
  exit 1
fi

cd /home/roberto/PythonProjects/MMDetectors/mmdetection

# get current date and time
now=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../checkpoints/"
OUTPUT_DIR+=$now


# MS3
VFNET_MS3_r50="configs/vfnet/vfnet_r50_fpn_1x_ms3.py"
VFNET_MS3_r18="configs/vfnet/vfnet_r18_fpn_1x_ms3.py"
RETINA_r18="configs/retinanet/retinanet_r50_fpn_1x_ms3.py"


# Set the configuration file to use:
# RUNTIME CONFIG
PROJECT_NAME="MS3"

MEANS="[200,154,116]"
STD="[22,24,27]"

# INPUTS CONFIG
MAX_EPOCHS=120
CHECKPOINT_RESUME="" # Set checkpoint resume to empty to start from scratch.


# RUN for different configurations:
for SEED in 1234
do
    for CONFIG_FILE in $RETINA_r18
    do
        model_name=$(basename "$CONFIG_FILE" | sed 's/\.py$//')
        # cycle for different values of the learning rate:
        for LR in 0.001
        do
            for BATCH_SIZE in 4
            do
                for IMG_SIZE in 1024
                do
                    now=$(date +"%Y%m%d_%H%M%S")
                    OUTPUT_DIR="../checkpoints/$PROJECT_NAME/$model_name/${now}_LR_${LR}_BATCH_${BATCH_SIZE}_IMG_${IMG_SIZE}_MEAN_${MEANS}_STD_${STD}"
                    echo $OUTPUT_DIR
                    # Run the train.py script
                    # if CHECKPOINT_RESUME is not empty, resume training from the checkpoint:
                    if [ -z "$CHECKPOINT_RESUME" ]
                    then
                        echo "CHECKPOINT_RESUME is empty"
                    else
                        echo "CHECKPOINT_RESUME is NOT empty"
                        echo $CHECKPOINT_RESUME
                    fi   
                    echo "Executing training ..." 
                    python tools/train.py ${CONFIG_FILE} \
                    --work-dir ${OUTPUT_DIR} \
                    --cfg-options randomness.seed=${SEED} \
                    optim_wrapper.optimizer.lr=${LR} \
                    train_dataloader.batch_size=${BATCH_SIZE} \
                    train_cfg.max_epochs=${MAX_EPOCHS} \
                    --resume ${CHECKPOINT_RESUME} \
                    --run_name ${model_name} \
                    --project_name ${PROJECT_NAME} \
                    --mean_norm_vals ${MEANS} \
                    --std_norm_vals ${STD}  \
                    --img_size ${IMG_SIZE} 

                    
                    # Filter checkpoints:
                    echo "Filtering checkpoints ..."
                    python tools/manage_checkpoints.py --workdir ${OUTPUT_DIR} --custom_mode="--custom_ms3"
                done
            done
        done
    done
done





# if keyboard interrupt, go back to the original directory
# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT
cd /home/roberto/PythonProjects/S2RAWVessel