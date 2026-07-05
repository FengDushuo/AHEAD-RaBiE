Atom-resolved ΔE_H* descriptor extraction patch

Files:
- 3g_detailed_extract_deltaeh_descriptor_joint_v3_atomresolved.py: extraction script with separate H-binding O/anchor-site and neighbouring-metal/surface descriptors.
- 4_build_hads_normalized_database_atomresolved.py: Step4 database builder with atom-resolved columns.
- run_pdf_files_add_deltaeh_descriptors_atomresolved_merge.sh: run script that calls the v3 extractor and writes to atomresolved output directories.

Important: if your existing run script calls 4_build_hads_normalized_database.sh, either replace 4_build_hads_normalized_database.py with the atomresolved version, or edit the shell script to call 4_build_hads_normalized_database_atomresolved.py directly.
