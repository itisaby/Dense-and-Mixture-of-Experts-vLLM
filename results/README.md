# Generated measurements

This directory is intentionally shipped without measured numbers. The serving
runner writes raw throughput CSVs, metadata, and logs here. The SFT runner
writes `sft/metrics.csv`, `sft/loss_history.csv`, fixed-selection metadata,
and the unedited held-out generations. Their corresponding plotting scripts
create summary tables.

Do not fill these files with estimated or illustrative values: the assignment
asks for empirical measurements.
