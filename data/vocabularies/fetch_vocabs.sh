#!/bin/bash

set -euo pipefail

BASE_DIR="${1:-$(pwd)}"
VOCABS=(
    "aat"
    "tgn"
    "ulan"
    "isni"
    "periodo"
    "fish_archaeological_objects"
    "fish_event_types"
    "fish_building_materials"
    "fish_monument_types"
    "bm_materials"
    "he_periods"
    "fish_subjects"
    "gbif_rank"
    "gbif_type_status"
    "dcmi_type"
    "iana_media_types"
    "loc_relators"
    "geonames"
    "os_open_names"
)

# Create all target directories
for vocab in "${VOCABS[@]}"; do
    mkdir -p "$BASE_DIR/$vocab"
done

download_vocab() {
    local vocab="$1"
    local url="$2"
    local filename="$3"
    local dir="$BASE_DIR/$vocab/$filename"

    echo "Downloading $vocab"
    curl -L -f -# -o "$dir" "$url" || echo "error downloading $vocab"
    echo "$vocab saved to $dir"
}

# Getty Vocabularies (AAT, TGN, ULAN)
for vocab in "aat" "tgn" "ulan"; do
    zip_name="${vocab}_data.zip"
    dir="$BASE_DIR/$vocab"
    zip_path="$dir/$zip_name"
    url="http://${vocab}downloads.getty.edu/VocabData/explicit.zip"

    download_vocab "$vocab" "$url" "$zip_name"
    unzip -o -q "$zip_path" -d "$dir"
    rm "$zip_path"
done

# ISNI public export, kept gzipped: the RDF/XML expands by roughly an order of
# magnitude and every reader can take it through gzip
ISNI_BASE="https://isni.oclc.org:2443/isni/public_export"
for export in persons organizations; do
    download_vocab "isni" "$ISNI_BASE/ISNI_${export}.rdf.gz" "ISNI_${export}.rdf.gz"
done

# Periodo
url_periodo="https://n2t.net/ark:/99152/p0dataset.json"
download_vocab "periodo" "$url_periodo" "periodo-dataset.json"

# FISH Vocabularies
FISH_BASE="https://heritage-standards.org.uk/2026/rdf_files"

url_fish_ao="FISH_Archaeological_Objects_20260204_Full.rdf"
download_vocab "fish_archaeological_objects" "${FISH_BASE}/$url_fish_ao" "$url_fish_ao"

url_fish_et="FISH_Event_Type_20260205.rdf"
download_vocab "fish_event_types" "${FISH_BASE}/$url_fish_et" "$url_fish_et"

url_fish_bm="Building_Materials_20260209.rdf"
download_vocab "fish_building_materials" "${FISH_BASE}/$url_fish_bm" "$url_fish_bm"

# FISH Thesaurus of Monument Types
zip_mt="MonumentTypeV29.zip"
download_vocab "fish_monument_types" \
    "https://heritage-standards.org.uk/2026/zip_files/$zip_mt" "$zip_mt"
unzip -o -q "$BASE_DIR/fish_monument_types/$zip_mt" -d "$BASE_DIR/fish_monument_types"
rm "$BASE_DIR/fish_monument_types/$zip_mt"

# Historic England Periods list
url_he_periods="HE_Periods1_20260408.csv"
download_vocab "he_periods" "https://heritage-standards.org.uk/2026/csv_files/$url_he_periods" "$url_he_periods"

# FISH Heritage Subjects & Themes
url_fish_subjects="http://purl.org/heritagedata/schemes/595"
download_vocab "fish_subjects" \
    "https://www.heritagedata.org/live/services/getConceptsForScheme?schemeURI=$url_fish_subjects" \
    "fish_subjects_concepts.json"

# British Museum Materials Thesaurus
BM_BASE="https://terminology.collectionstrust.org.uk/British-Museum-materials"
download_vocab "bm_materials" "$BM_BASE/matintro.htm" "matintro.htm"
for letter in a b c d e f g h i j k l m n o p q r s t u v w y z; do
    download_vocab "bm_materials" "$BM_BASE/mathes${letter}.htm" "mathes${letter}.htm"
    download_vocab "bm_materials" "$BM_BASE/maindex${letter}.htm" "maindex${letter}.htm"
done

# GBIF thesauri for the natural-science fields
GBIF_BASE="https://rs.gbif.org/vocabulary/gbif"
download_vocab "gbif_rank" "$GBIF_BASE/rank.xml" "rank.xml"
download_vocab "gbif_type_status" "$GBIF_BASE/type_status.xml" "type_status.xml"

# DCMI Type Vocabulary
download_vocab "dcmi_type" \
    "https://www.dublincore.org/specifications/dublin-core/dcmi-terms/dublin_core_type.ttl" \
    "dublin_core_type.ttl"

# IANA media type registry
IANA_BASE="https://www.iana.org/assignments/media-types"
for registry in image audio video application text; do
    download_vocab "iana_media_types" "$IANA_BASE/$registry.csv" "$registry.csv"
done

# Library of Congress MARC Relator terms
download_vocab "loc_relators" "https://www.loc.gov/marc/relators/relaterm.html" "relaterm.html"

# GeoNames
GEONAMES_BASE="https://download.geonames.org/export/dump"
for archive in allCountries alternateNamesV2; do
    download_vocab "geonames" "$GEONAMES_BASE/${archive}.zip" "${archive}.zip"
    unzip -o -q "$BASE_DIR/geonames/${archive}.zip" -d "$BASE_DIR/geonames"
    rm "$BASE_DIR/geonames/${archive}.zip"
done
for table in admin1CodesASCII.txt admin2Codes.txt countryInfo.txt featureCodes_en.txt; do
    download_vocab "geonames" "$GEONAMES_BASE/$table" "$table"
done

# OS Open Names
download_vocab "os_open_names" \
    "https://api.os.uk/downloads/v1/products/OpenNames/downloads?area=GB&format=CSV&redirect" \
    "opname_csv_gb.zip"
unzip -o -q "$BASE_DIR/os_open_names/opname_csv_gb.zip" -d "$BASE_DIR/os_open_names"
rm "$BASE_DIR/os_open_names/opname_csv_gb.zip"