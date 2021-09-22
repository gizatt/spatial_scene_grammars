# Utility for unzipping all zip files in directories under the first
# argument into folders with the same name as the zip file, alongside
# wherever the zip file is.
find $1 -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

