set -ex

cathedral_number=$1
feature_type=$2
colmap_number=$3

nerfstudio_cathderals=./local/nerfstudio_data/cathedrals/
nerfstudio_cathedral_dir=$nerfstudio_cathderals/$cathedral_number/registered_meta_image/$colmap_number

./scripts/features/create_features.sh $nerfstudio_cathedral_dir $feature_type

ln -s -f $(realpath $nerfstudio_cathedral_dir/images_${feature_type}_features) $nerfstudio_cathderals/$cathedral_number/wild_registration/$colmap_number/images_${feature_type}_features

gt_dir=$nerfstudio_cathderals/$cathedral_number/gt

if [ -d $gt_dir ]; then
    ln -s -f $(realpath $nerfstudio_cathedral_dir/images_${feature_type}_features) $gt_dir/images_${feature_type}_features
else
    echo "GT dir does not exist"
fi