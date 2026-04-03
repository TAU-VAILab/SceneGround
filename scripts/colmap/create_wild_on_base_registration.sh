
set -ex
cathedral_number=$1
colmap_number=$2
ignore_gt=$3

echo "[INFO] cathedral_number: $cathedral_number - creating wild on base registration"
echo "using common custom parameters"

# Set meta_image_cathedral_number based on cathedral_number
if [ $cathedral_number -gt 200 ]; then
    meta_image_cathedral_number=$(($cathedral_number - 200))
else
    meta_image_cathedral_number=$cathedral_number
fi

subdir="wild_registration/$colmap_number"
cathedral_colmap_models_dir=./local/colmap_models/$cathedral_number
wild_colmap_path="./local/colmap_models/$meta_image_cathedral_number/meta_image/$colmap_number/sparse/0"

database_path="$cathedral_colmap_models_dir/$subdir/database.db"
base_wikiscenes_images_dir=./datasets/WikiScenes1200px/WikiScenes1200px/cathedrals
base_sparse_path=$cathedral_colmap_models_dir/base/sparse/0
wild_colmap_on_earth=$cathedral_colmap_models_dir/$subdir/sparse
base_database_path=$cathedral_colmap_models_dir/base/database.db


# ----------- Creating the registered meta image colmap model ----------------

registered_meta_image_base=$cathedral_colmap_models_dir/registered_meta_image/$colmap_number

mkdir -p $wild_colmap_on_earth
mkdir -p $registered_meta_image_base/sparse

rm -f ./local/colmap_models/$cathedral_number/$subdir/database.db* 
cp $base_database_path ./local/colmap_models/$cathedral_number/$subdir/

echo "build_colmap_with_gt_intrinsics for: $cathedral_number, $subdir"


if [ $ignore_gt = "1" ]; then
    extra_flags="--no_ground_truth"
fi
#TODO for drone video should use --no_ground_truth
python ./scripts/colmap/build_colmap_with_gt_intrinsics.py $cathedral_number $colmap_number --on_google_earth $extra_flags

python ./scripts/create_geo_registration_file.py $wild_colmap_on_earth $registered_meta_image_base/geo_registration.txt 

colmap model_aligner --input_path $wild_colmap_path --output_path $registered_meta_image_base/sparse --alignment_max_error 3 --alignment_type custom --ref_is_gps 0 --ref_images_path $registered_meta_image_base/geo_registration.txt 2>&1 | grep "Alignment error:" > $registered_meta_image_base/alignment_error.txt

colmap model_analyzer --path $wild_colmap_on_earth 2> $registered_meta_image_base/wild_model_analysis.txt

echo "Finished Creating colmap model: $cathedral_number, $subdir"


# ----------- Creating the registered meta image nerfstudio directory ----------------

nerfstudio_cathderals=./local/nerfstudio_data/cathedrals/
nerfstudio_cathedral_dir=$nerfstudio_cathderals/$cathedral_number/registered_meta_image/$colmap_number
nerfstudio_colmap_path=$nerfstudio_cathedral_dir/sparse


# check if $cathedral_number is bigger then 100
if [ $cathedral_number -lt 101 ]; then
images_path=./datasets/WikiScenes1200px_Partial/Partial_$cathedral_number/datasets/WikiScenes1200px/WikiScenes1200px/cathedrals
elif [ $cathedral_number -lt 201 ]; then
images_path=./datasets/WikiScenes1200px/WikiScenes1200px/cathedrals/$cathedral_number
else
images_path=./datasets/WikiScenes1200px_Partial/Partial_$meta_image_cathedral_number/datasets/WikiScenes1200px/WikiScenes1200px/cathedrals
fi

echo mkdir -p $nerfstudio_cathedral_dir

mkdir -p $nerfstudio_colmap_path

cp $registered_meta_image_base/sparse/* $nerfstudio_colmap_path

ns-process-data images --data $images_path --skip-colmap --colmap-model-path ./sparse --output_dir $nerfstudio_cathedral_dir

# ----------- Creating wild registration colmap nerfstudio directory ----------------
nerfstudio_wild_cathedral_dir=$nerfstudio_cathderals/$cathedral_number/wild_registration/$colmap_number
nerfstudio_colmap_path=$nerfstudio_wild_cathedral_dir/sparse

mkdir -p $nerfstudio_colmap_path

cp $wild_colmap_on_earth/* $nerfstudio_colmap_path

ns-process-data images --data $images_path --skip-colmap --colmap-model-path ./sparse --output_dir $nerfstudio_wild_cathedral_dir