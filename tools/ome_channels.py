#!/usr/bin/env python3
"""Print the channel names and pixel geometry stored in an OME-TIFF / qptiff header.

Read-only: opens the file for reading, never writes to it. Works on .ome.tif, .qptiff and
.ome.tif.gz (for .gz it streams only the first --scan-mb megabytes and searches for the XML).

usage: python ome_channels.py FILE [FILE ...] [--scan-mb 64]
"""
import argparse, gzip, re, sys


def from_bytes(buf, name):
    txt = buf.decode("latin-1")
    found = False
    m = re.search(r"<OME[\s>].*?</OME>", txt, re.S)
    if m:
        found = True
        xml = m.group(0)
        px = re.search(r"<Pixels\b[^>]*>", xml)
        if px:
            kv = re.findall(r'(Size[XYZCT]|PhysicalSize[XY]|Type)="([^"]*)"', px.group(0))
            print("  Pixels:", " ".join(f"{k}={v}" for k, v in kv))
        for i, ch in enumerate(re.findall(r"<Channel\b[^>]*/?>", xml)):
            attrs = dict(re.findall(r'(\w+)="([^"]*)"', ch))
            keep = {k: v for k, v in attrs.items() if k in ("ID", "Name", "Fluor", "EmissionWavelength",
                                                             "ExcitationWavelength", "Color")}
            print(f"  channel {i}: {keep}")
        for k in ("AnnotationRef", "MapAnnotation", "XMLAnnotation"):
            if k in xml:
                print(f"  (OME-XML also holds {k} entries; marker names may sit there)")
                break
    # Akoya qptiff: one <PerkinElmer-QPI-ImageDescription> per page, with <Name> and <Biomarker>
    q = re.findall(r"<PerkinElmer-QPI-ImageDescription>.*?</PerkinElmer-QPI-ImageDescription>", txt, re.S)
    if q:
        found = True
        for i, block in enumerate(q):
            nm = re.search(r"<Name>(.*?)</Name>", block)
            bm = re.search(r"<Biomarker>(.*?)</Biomarker>", block)
            print(f"  qptiff page {i}: name={nm.group(1) if nm else '?'} biomarker={bm.group(1) if bm else '?'}")
    if not found:
        print("  no OME-XML or qptiff description found in the scanned part;"
              " try a larger --scan-mb, or the decompressed file")
    hits = sorted(set(re.findall(r"(?i)\b(opal\s*\d{3}|cd8a?|pan-?ck|pan-?cytokeratin|dapi|"
                                 r"autofluorescence|sample af)\b", txt)))
    if hits:
        print("  marker/fluor words anywhere in header:", ", ".join(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scan-mb", type=int, default=64)
    a = ap.parse_args()
    for f in a.files:
        print(f"== {f}")
        n = a.scan_mb * 1024 * 1024
        try:
            if f.endswith(".gz"):
                with gzip.open(f, "rb") as fh:
                    buf = fh.read(n)
            else:
                try:
                    import tifffile
                    with tifffile.TiffFile(f) as tf:
                        print(f"  pages: {len(tf.pages)}; series shapes: {[s.shape for s in tf.series]}")
                        xml = tf.ome_metadata or ""
                        descs = [p.description for p in tf.pages[:64] if getattr(p, "description", None)]
                        buf = (xml + "\n".join(descs)).encode("latin-1", "replace")
                except ImportError:
                    with open(f, "rb") as fh:
                        buf = fh.read(n)
            from_bytes(buf, f)
        except Exception as e:  # report and continue with the next file
            print(f"  could not read: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
