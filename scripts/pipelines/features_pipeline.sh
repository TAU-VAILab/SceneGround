set -ex

cathedral_number=$1
feature_type=$2
colmap_number=$3
near_plane=$4
loss_percentile=$5
angle=$6
translation=$7
extra=$8
image_percent=$9

#initializations options
#initialization=registered_super_glue
#initialization=gpnps
#initialization=super_glue

# if angle is not define set to 0
if [ -z "$angle" ]; then
    angle=0
fi
# if translation is not define set to 0
if [ -z "$translation" ]; then
    translation=0
fi

# If near_plane is not define set to 0.7
if [ -z "$near_plane" ]; then
    near_plane=0.7
fi

if [ -z "$loss_percentile" ]; then
    loss_percentile=50
fi

# loss_masking_percentile Deprecated
loss_masking_percentile=100
iteration_number=12000


transform_path=./transforms/$cathedral_number/$colmap_number/percentile-${loss_percentile}_iteration-${iteration_number}_near-plane-${near_plane}__l2_loss-False_feature_type-${feature_type}_loss_masking-${loss_masking_percentile}_transform.pt

if [ "$feature_type" != "rgb" ]; then
    echo "Creating Features"
    ./scripts/features/create_all_features.sh $cathedral_number $feature_type $colmap_number
else
    echo "Feature type is rgb, skipping feature creation"
fi

# increase iteration_number if increasing the loss_percentile
if [ -z "$extra" ]; then
    extra_param=""
else
    extra_param="--extra $extra"
fi
line_to_exec="python ./scripts/registration/feature_registration.py single $cathedral_number $colmap_number --feature_type $feature_type --iteration_number $iteration_number --loss_percentile $loss_percentile --near_plane $near_plane --loss_masking $loss_masking_percentile --angle $angle --translation $translation $extra_param"
# if image_percent is not defined, add the flag to the line_to_exec
if [ -n "$image_percent" ]; then
    line_to_exec="$line_to_exec --image_percent $image_percent"
fi

if [ -n "$initialization" ]; then
    line_to_exec="$line_to_exec --initialization $initialization"
    
    nerfstudio_cathderals=./local/nerfstudio_data/cathedrals/
    nerfstudio_cathedral_dir=$nerfstudio_cathderals/$cathedral_number/$initialization/$colmap_number
    if [ ! -f $nerfstudio_cathedral_dir/transforms.json ]; then
        scripts/colmap/create_initialization_ns.sh  $cathedral_number $colmap_number $feature_type $initialization
    fi
fi
# exec the line
$line_to_exec


# mkdir -p ./results/transforms/$cathedral_number/

# echo "=============Results============="
# python ./scripts/compare_registration_to_gt.py single-cathedrals $cathedral_number $colmap_number --transform_path $transform_path

# Visualizations
# Disabling visualizations for now
# cp $transform_path ./results/transforms/$cathedral_number/
# python ./scripts/visualize_registration.py single $cathedral_number $colmap_number $feature_type --render_anyway
