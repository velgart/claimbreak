# Changelog

## 1.7.103 (first public release)

First public release. Version numbers before 1.7.103 were internal development versions. The results of the
accompanying paper were produced with the same computations; the example scripts in `examples/` are the ones behind
its analyses. Example fixes: `vendor_calls.py` stops with a message, instead of a KeyError, when no field of view
reaches `--min-t` and `--min-p`; `ihc_channel_check.py` and `resolvi_rcc.py` also run on Python 3.10 and 3.11.
