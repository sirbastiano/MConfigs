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
cd /home/roberto/PythonProjects/S2RAWVessel/mmdetection

# get current date and time
now=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="../checkpoints/"
OUTPUT_DIR+=$now


# BASE CONFIGURATION FILES:
##################################################################
RETINA_R50="configs/retinanet/retinanet_r50_fpn_1x_vessel.py"
RETINA_R18="configs/retinanet/retinanet_r18_fpn_1x_vessels.py"
FASTER_R50="configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco_vessels.py"
VFNET_R18="configs/vfnet/vfnet_r18_fpn_1x_vessel.py"
VFNET_R50="configs/vfnet/vfnet_r50_fpn_1x_vessel.py"
VFNET_R101="configs/vfnet/vfnet_r101_fpn_1x_vessel.py"
VFNET_EFFB3="configs/vfnet/vfnet_Effb3_fpn_1x_vessel.py"
VFNET_R18_PAFPN="configs/vfnet/vfnet_r18_pafpn_1x_vessel.py"
RTMDET_L="configs/rtmdet/rtmdet_l_8xb32-300e_vessel.py"
CASCADE_MASK_R50="configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_240_vessel.py"
YOLOV3="configs/yolo/yolov3_d53_8xb8-ms-608-273e_vessel.py"
FOVEABOX="configs/foveabox/fovea_r50_fpn_4xb4-1x_vessel.py"
YOLOX_S="configs/yolox/yolox_s_8xb8-300e_vessel.py"
YOLOX_tiny="configs/yolox/yolox_tiny_8xb8-300e_vessel.py"
YOLOX_nano="configs/yolox/yolox_nano_8xb8-300e_vessel.py"
RETINA_EFF="configs/efficientnet/retinanet_effb3_fpn_8xb4-crop896-1x_vessel.py"
RTMDET_S="configs/rtmdet/rtmdet_s_8xb32-300e_vessel.py"
FCOS_R18="configs/fcos/fcos_r18_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_vessel.py"
NAS_FPN_R18="configs/nas_fpn/retinanet_r18_fpn_crop640-50e_vessel.py"
YOLOX_tiny_boost="configs/yolox/yolox_tiny_8xb8-300e_vessel_scaleBoost.py"
SSD="configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_vessel.py"
CENTERNET="configs/centernet/centernet-update_r50_fpn_8xb8-amp-lsj-200e_vessel.py"
RETINA_REGNETX_32F="configs/regnet/retinanet_regnetx-3.2GF_fpn_1x_vessel.py"
DETR="configs/detr/detr_r50_8xb2-150e_vessel.py"
FSAF="configs/fsaf/fsaf_r50_fpn_1x_vessel.py"
RETINA_SWIN="configs/swin/retinanet_swin-t-p4-w7_fpn_1x_vessel.py"
DINO_R50="/home/roberto/PythonProjects/S2RAWVessel/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_vessel.py"
YOLOX_nano_test="/configs/yolox/yolox_nano_8xb8-300e_vessel_test.py"
DOUBLE_HEADS_R50="configs/double_heads/dh-faster-rcnn_r50_fpn_1x_vessel.py"
CENTRIPETALNET_HOURGLASS104="configs/centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_vessel.py"
##################################################################
# MS3
VFNET_MS3_r50="configs/vfnet/vfnet_r50_fpn_1x_ms3.py"
##################################################################
# ESA
VFNET_ESA_r50="configs/vfnet/vfnet_r50_fpn_1x_vessel_esa.py"



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
    for CONFIG_FILE in $VFNET_MS3_r50
    do
        model_name=$(basename "$CONFIG_FILE" | sed 's/\.py$//')
        # cycle for different values of the learning rate:
        for LR in 0.0001
        do
            for BATCH_SIZE in 6
            do
                for IMG_SIZE in 2048
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
                    python tools/manage_checkpoints.py --workdir ${OUTPUT_DIR}
                done
            done
        done
    done
done





# if keyboard interrupt, go back to the original directory
# Trap keyboard interrupt and call the interrupt_handler function
trap interrupt_handler SIGINT
cd /home/roberto/PythonProjects/S2RAWVessel