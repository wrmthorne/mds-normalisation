# notebooks

Narrative walkthroughs of the pipeline, one per stage. They import the library as `mds_norm.*` and read the artefacts under `data/analysis_output/`.

| Notebook           | Stage                                                         | Module                                               |
|--------------------|---------------------------------------------------------------|------------------------------------------------------|
| `1_field_analysis` | Tier-0 standardisation, per-value features, pattern induction | `pipeline.field_census`                              |
| `2_probe_scan`     | Structured content misfiled in free text                      | `pipeline.probe_scan`                                |
| `3_misplacement`   | Fields an institution maps the wrong content into             | `pipeline.misplacement_scan`                         |
| `4_vocabularies`   | Authority indexes and the alignment cascade                   | `pipeline.vocab_indexes`, `pipeline.vocab_alignment` |
| `5_entities`       | Agent names: type routing and name parsing                    | `parsers.parse_person`                               |
| `6_record_fixes`   | Record-level patch ops and the LLM extraction tier            | `pipeline.extraction`                                |
| `7_evaluation`     | Cost per stage against quality gained                         | `evaluation.reports`, `evaluation.quality_deltas`    |
