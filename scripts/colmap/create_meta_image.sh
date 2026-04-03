set -ex
cathedral_number=$1
colmap_number=$2
ignore_gt=$3

ignore_gt=${ignore_gt:-0}

echo "[INFO] cathedral_number: $cathedral_number - creating wild on base registration"
database_path=./local/colmap_models/$cathedral_number/meta_image/$colmap_number/database.db
rm -f $database_path*
rm -rf ./local/colmap_models/$cathedral_number/meta_image/$colmap_number/sparse/

if [ $ignore_gt -eq 1 ]; then
    original_colmap_model=./datasets/WikiScenes3D/$cathedral_number/$colmap_number
    dest_colmap_dir=./local/colmap_models/$cathedral_number/meta_image/$colmap_number/sparse/0
    echo "Skipping creation of colmap model"
    mkdir -p $dest_colmap_dir
    cp -r $original_colmap_model/* $dest_colmap_dir
else
    python ./scripts/colmap/build_colmap_with_gt_intrinsics.py $cathedral_number $colmap_number
fi


