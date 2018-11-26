#!/usr/bin/env bash

TIMESTAMP=`date "+%Y-%m-%d %H:%M:%S"`

declare -a CHECKPOINT_URLS=(
"http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
"http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz"
)

usage() {
    echo "This script downloads and extracts .ckpt files from the following URLs:"
    for CHECKPOINT_URL in ${CHECKPOINT_URLS[@]}; do
        echo "  - ${CHECKPOINT_URL}"
    done
    echo ""
    echo "USAGE:"
    echo "\$ sh get_model_checkpoints.sh /path/to/checkpoints"
    echo ""
}

download() {
    local DOWNLOAD_URL="$1"
    local TARGET_DIR=$(cd "$2" && pwd)

    echo "[${TIMESTAMP}] Downloading ${DOWNLOAD_URL} to ${TARGET_DIR}"

    mkdir -p ${TARGET_DIR}
    wget -q --show-progress ${DOWNLOAD_URL} -P ${TARGET_DIR}
}

untar_and_cleanup() {
    local TARGET_DIR=$(cd $(dirname "$1") && pwd)
    local TARGET_FILEPATH="$1"

    echo "[${TIMESTAMP}] Extracting checkpoint file from $1"

    tar -C ${TARGET_DIR} -xzf ${TARGET_FILEPATH}
    rm -f ${TARGET_FILEPATH}
    rm -f ${TARGET_DIR}/*.graph
}

if [  $# -ne 1 ]; then
    usage
    echo "ERROR: need path to checkpoints directory"
    echo ""
    exit 1
fi

if [[ ( $1 == "--help") ||  $1 == "-h" ]]; then
    usage
    exit 0
fi

if [ ! -d "$1" ]; then
    usage
    echo "ERROR: $1 does not exist"
    echo ""
    exit 1
fi

declare -r CHECKPOINT_DIR=$(cd "$1" && pwd)

for CHECKPOINT_URL in ${CHECKPOINT_URLS[@]}; do
    DOWNLOAD_FILENAME=$(basename ${CHECKPOINT_URL})

    download "${CHECKPOINT_URL}" "${CHECKPOINT_DIR}"
    untar_and_cleanup "${CHECKPOINT_DIR}/${DOWNLOAD_FILENAME}"
done

echo "[${TIMESTAMP}] Finished downloading checkpoint files"
