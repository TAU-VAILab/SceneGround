set -ex
nerfstudio_cathedral_dir=$1
feature_type=$2
feature_extractor_path=./feature_extractor/extract_feature_script.py

python $feature_extractor_path cathedral $nerfstudio_cathedral_dir $feature_type -v --skip_existing
