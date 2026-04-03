#!/bin/bash
set -ex

clone_datasets () {
    wget https://www.cs.cornell.edu/projects/babel/WikiScenes1200px.tar.gz
    mkdir -p ./datasets/WikiScenes1200px
    tar -xvf WikiScenes1200px.tar.gz -C ./datasets/WikiScenes1200px
    rm WikiScenes1200px.tar.gz

    wget https://www.cs.cornell.edu/projects/babel/WikiScenes3D.zip
    mkdir -p ./datasets/WikiScenes3D/
    unzip WikiScenes3D.zip -d ./datasets/WikiScenes3D
    rm WikiScenes3D.zip

    wget https://huggingface.co/datasets/cornell-vailab/WikiEarth/resolve/main/wikiearth_dataset.tar.gz
    mkdir -p ./datasets/wikiearth_dataset/
    tar -xvf wikiearth_dataset.tar.gz
    rm wikiearth_dataset.tar.gz

}

create_wikiscenes_1200px_partial() {
    # Define the base directories
    source_base="./datasets/WikiScenes1200px/WikiScenes1200px/cathedrals/"
    target_base="./datasets/WikiScenes1200px_Partial"

    # # Iterate over directories in the source base directory
    for dir in "$source_base"/*/; do
        # Extract the directory name (number)
        dir_name=$(basename "$dir")
        
        # Create the target directory if it doesn't exist
        target_dir="$target_base/Partial_$dir_name"
        echo "creating dir: $target_dir"

        mkdir -p "$target_dir"
        
        cathedral_wikiscenes="$source_base/$dir_name"

        cathedral_target_dir=$target_dir/$source_base/

        mkdir -p $cathedral_target_dir
        echo "making dir: $cathedral_target_dir"
        # Create the symbolic link
        cp -r "$cathedral_wikiscenes" "$cathedral_target_dir/$dir_name"
        
        echo "cp -r: $cathedral_wikiscenes $cathedral_target_dir/$dir_name"
    done
}

convert_wild_colmaps_to_bin() {
    echo "Converting wild colmap models to BIN format"
    # Base directory
    base_dir="datasets/WikiScenes3D"

    # Iterate over each cathedral_number directory
    for cathedral_dir in "$base_dir"/*; do
        if [ -d "$cathedral_dir" ]; then
            cathedral_number=$(basename "$cathedral_dir")
            
            # Iterate over each colmap_number directory within the current cathedral_number directory
            for colmap_dir in "$cathedral_dir"/*; do
                if [ -d "$colmap_dir" ]; then
                    colmap_number=$(basename "$colmap_dir")
                    if [ -n "$colmap_number" ]; then
                        # Execute the colmap model_converter command
                        colmap model_converter --input_path "$colmap_dir" --output_path "$colmap_dir" --output_type BIN 2> /dev/null
                        echo "Converting $colmap_dir to BIN format"
                    fi
                fi
            done
        fi
    done
}

# Initial setup
clone_datasets
create_wikiscenes_1200px_partial
convert_wild_colmaps_to_bin
