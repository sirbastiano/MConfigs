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

##################################################################
VFNET_ESA_r50="configs/vfnet/vfnet_r50_fpn_1x_vessel_esa.py"
VFNET_ESA_r18="configs/vfnet/vfnet_r18_fpn_1x_esa.py"
VFNET_ESA_r18_P="configs/vfnet/vfnet_r18_fpn_1x_esa_playground.py"
VFNET_ESA_r101="configs/vfnet/vfnet_r101_fpn_1x_esa.py"
VFNET_ESA_RES2NET101="configs/vfnet/vfnet_res2net-101_fpn_ms-2x_esa.py"
VFNET_ESA_RESNEXT101="configs/vfnet/vfnet_x101-64x4d_fpn_ms-2x_esa.py"
VFNET_HRNET_ESA="configs/vfnet/vfnet_hrnet_esa.py"
VFNET_ESA_EFF_B3="configs/vfnet/vfnet_Effb3_fpn_1x_esa.py"

CASCADE_MASK_RCNN_R50="configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_S2L1C.py"
YOLOX_S="configs/yolox/yolox_s_8xb8-300e_S2L1C.py"
FOVEABOX="configs/foveabox/fovea_r50_fpn_4xb4-1x_esa.py"
##################################################################

# Set the configuration file to use:
# RUNTIME CONFIG
PROJECT_NAME="S2L1C_1024"

MEANS="[50,50,50]"
STD="[50,50,50]"

# INPUTS CONFIG
CHECKPOINT_RESUME="" # Set checkpoint resume to empty to start from scratch.
MAX_EPOCHS=15

# RUN for different configurations:
for STD in "[100,100,100]" 
do
    for MEANS in "[50,50,50]" 
    do
        for SEED in 3142
        do
            for CONFIG_FILE in $CASCADE_MASK_RCNN_R50
            do
                model_name=$(basename "$CONFIG_FILE" | sed 's/\.py$//')
                # cycle for different values of the learning rate:
                for LR in 0.0007 0.0008 0.0009
                do
                    for BATCH_SIZE in 12
                    do
                        for IMG_SIZE in 1024
                        do
                            now=$(date +"%Y%m%d_%H%M%S")
                            OUTPUT_DIR="../checkpoints/$PROJECT_NAME/norm_test_$model_name/${now}_LR_${LR}_BATCH_${BATCH_SIZE}_IMG_${IMG_SIZE}_MEAN_${MEANS}_STD_${STD}"
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
                            # --img_size ${IMG_SIZE} 

                            
                            # Filter checkpoints:
                            echo "Filtering checkpoints ..."
                            python tools/manage_checkpoints.py --workdir ${OUTPUT_DIR} --custom_mode="--custom_s2l1c"
                        done
                    done
                done
            done
        done
    done
done





# if keyboard interrupt, go back to the original directory
# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT
cd /home/roberto/PythonProjects/S2RAWVessel