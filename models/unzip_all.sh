# Utility for unzipping all zip files in directories under this
# one into folders with the same name as the zip file, alongside
# wherever the zip file is.
find . -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

