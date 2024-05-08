
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
vfnet_r50="configs/vfnet/vfnet_r50_fpn_1x_venus.py"
vfnet_r18="configs/vfnet/vfnet_r18_fpn_1x_venus.py"

faster_r50="configs/faster_rcnn/faster-rcnn_r50_fpn_1x_venus.py"
retina_r18="configs/retinanet/retinanet_r18_fpn_1x_venus.py"

yolox_s="configs/yolox/yolox_s_8xb8-300e_venus.py"

##################################################################

# Set the configuration file to use:
# RUNTIME CONFIG
PROJECT_NAME="Venus"

MEANS="[158.69588,124.42161,109.27108,105.380424,88.40926,98.93067,88.819916,94.20678,103.540764,111.64337,122.92817,79.31501]"
STD="[34.95446,46.282494,56.252197,55.741932,64.54027,59.59095,69.65824,68.40028,77.930405,103.4634,105.30468,65.8369]"


# INPUTS CONFIG
CHECKPOINT_RESUME="" # Set checkpoint resume to empty to start from scratch.
MAX_EPOCHS=15

# RUN for different configurations:
for STD in "[34.95446,46.282494,56.252197,55.741932,64.54027,59.59095,69.65824,68.40028,77.930405,103.4634,105.30468,65.8369]"
do
    for MEANS in "[158.69588,124.42161,109.27108,105.380424,88.40926,98.93067,88.819916,94.20678,103.540764,111.64337,122.92817,79.31501]"
    do
        for SEED in 1 2 3 
        do
            for CONFIG_FILE in  $retina_r18 $vfnet_r50 $vfnet_r18 $faster_r50
            do
                model_name=$(basename "$CONFIG_FILE" | sed 's/\.py$//')
                # cycle for different values of the learning rate:
                for LR in 0.05 0.001 0.0015  
                do
                    for BATCH_SIZE in 4 6 8 
                    do
                        for IMG_SIZE in 2304
                        do
                            now=$(date +"%Y%m%d_%H%M%S")
                            OUTPUT_DIR="../checkpoints/$PROJECT_NAME/norm_test_$model_name/${now}_LR_${LR}_BATCH_${BATCH_SIZE}_IMG_${IMG_SIZE}"
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

                            
                            # Filter checkpoints:
                            echo "Filtering checkpoints ..."
                            python tools/manage_checkpoints.py --workdir ${OUTPUT_DIR} --custom_mode="--custom_venus"
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



