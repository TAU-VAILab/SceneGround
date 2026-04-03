set -ex
cathedral_number=$1
features_type=$2


dst_base="./local/nerfstudio_data/cathedrals"
model_dir="$dst_base/$cathedral_number/base_google_earth"
random_string=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13)
experiment_name="splatfacto_features_$random_string"
model_out_dir=./local/model_checkpoints/cathedrals/$cathedral_number/base_google_earth/features_oriented_$features_type/
step_number=12000
output_model_dir="outputs/$experiment_name/splatfacto"

if [ -d $model_out_dir/nerfstudio_models ]; then
    echo "skipping $cathedral_number becuase it already exists"
    exit 0
fi

./scripts/features/create_features.sh $model_dir $features_type

ns-train splatfacto_features --experiment_name $experiment_name --pipeline.model.feature_type $features_type --data $model_dir --max_num_iterations $step_number --viewer.quit-on-train-completion True --vis viewer+tensorboard nerfstudio-data --auto_scale_poses False --feature-type $features_type

model_dir_name=$(ls $output_model_dir | tail -n 1)

mkdir -p ./local/model_checkpoints/cathedrals/$cathedral_number/base_google_earth/features_oriented_$features_type/
cp -r $output_model_dir/$model_dir_name/* $model_out_dir