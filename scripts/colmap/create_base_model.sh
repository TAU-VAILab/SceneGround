set -ex

dst_base="./local/nerfstudio_data/cathedrals"

cathedral_number=$1
images_path=./datasets/wikiearth_dataset/$cathedral_number/GoogleEarthRenders/images

create_base_model() {
   mkdir -p ./local/colmap_models/$cathedral_number/base/sparse

   colmap feature_extractor \
      --database_path ./local/colmap_models/$cathedral_number/base/database.db \
      --image_path $images_path

   #colmap sequential_matcher - for drone video
   colmap spatial_matcher \
      --database_path ./local/colmap_models/$cathedral_number/base/database.db

   colmap mapper \
      --database_path ./local/colmap_models/$cathedral_number/base/database.db \
      --image_path $images_path \
      --output_path ./local/colmap_models/$cathedral_number/base/sparse \
      --Mapper.ignore_watermarks 1
}

create_base_nerf_studio_model() {
   model_dir="$dst_base/$cathedral_number/base_google_earth"
   mkdir -p $model_dir
   cp -r ./local/colmap_models/$cathedral_number/base/sparse/0 $model_dir/sparse
   ns-process-data images --data $images_path --skip-colmap --colmap-model-path ./sparse --output_dir $model_dir
}

echo "creating base model for $cathedral_number"
create_base_model $cathedral_number
echo "Finished: $cathedral_number, create_base_model"
create_base_nerf_studio_model $cathedral_number
echo "Finished: $cathedral_number, create_base_nerf_studio_model"
