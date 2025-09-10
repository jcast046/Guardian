# Guardian
Guardian is a lightweight, AI-driven OSINT system focused on consolidating and analyzing reports of missing women and children.  The purpose of this project is to build a proof-of-concept tool that demonstrates how natural language processing (NLP), unsupervised machine learning, and geospatial visualization can be applied to identify trends, patterns, and hotspots to more directly assist in locating victims in missing person cases.

# -> [Guardian_Parser](https://github.com/jcast046/Guardian_parser) <-


The **Guardian Parser** is a pipeline for converting unstructured missing person case PDFs (from **NamUs**, **NCMEC**, and **The Charley Project**) into a unified, structured format based on the **Guardian JSON Schema**. It extracts demographic, spatial, temporal, outcome, and narrative/OSINT fields, normalizes them, and outputs both JSONL and CSV files for downstream analysis and synthetic data generation for this project.
