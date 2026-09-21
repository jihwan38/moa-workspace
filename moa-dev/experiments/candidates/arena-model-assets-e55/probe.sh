#!/bin/sh
# Bounded metadata probe in our Arena job. Never opens model or credential bytes.
set -eu

check_file() {
    probe_regular=0
    probe_readable=0
    [ ! -f "$2" ] || probe_regular=1
    [ ! -r "$2" ] || probe_readable=1
    printf 'file\t%s\t%s\t%s\n' "$1" "$probe_regular" "$probe_readable"
}

check_directory() {
    check_file "$1/model.safetensors" "$2/model.safetensors"
    check_file "$1/tokenizer.json" "$2/tokenizer.json"
    check_file "$1/tokenizer_config.json" "$2/tokenizer_config.json"
    check_file "$1/chat_template.jinja" "$2/chat_template.jinja"
    check_file "$1/chat_template.json" "$2/chat_template.json"
}

check_root() {
    check_directory "$1/direct" "$2"
    check_directory "$1/Gemma4-12B-it" "$2/Gemma4-12B-it"
    check_directory "$1/gemma-4-12b-it-NVFP4" "$2/gemma-4-12b-it-NVFP4"
}

printf 'MOA_MODEL_ASSET_PRESENCE_V1\n'
if [ "${RNGD_MODEL_DIR+x}" = x ]; then
    printf 'RNGD_MODEL_DIR\tset\n'
    # Empty configured path has the current-directory meaning used by PathBuf.
    check_root configured "${RNGD_MODEL_DIR:-.}"
else
    printf 'RNGD_MODEL_DIR\tunset\n'
fi

# These conventional paths are bounded fallbacks, not documented model mounts.
# No recursion, directory enumeration, environment dump, or file-content reads.
check_root models /models
check_root data /data
check_root workspace /workspace
printf 'MOA_MODEL_ASSET_PRESENCE_END\n'
