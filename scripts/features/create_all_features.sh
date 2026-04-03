set -ex

cathedral_number=$1
feature_type=$2
colmap_number=$3

./scripts/features/create_features_model.sh $cathedral_number $feature_type
./scripts/features/create_wild_features.sh $cathedral_number $feature_type $colmap_number