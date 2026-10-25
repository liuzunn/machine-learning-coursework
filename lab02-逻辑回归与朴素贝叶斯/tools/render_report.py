# -*- coding: utf-8 -*-
import fitz, sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
src = r"C:\Users\LZ\Desktop\机器学习实验一\_report_preview.pdf"
outdir = r"C:\Users\LZ\Desktop\机器学习实验一\_preview"
os.makedirs(outdir, exist_ok=True)
doc = fitz.open(src)
print("pages", doc.page_count)
for i in range(doc.page_count):
    pix = doc[i].get_pixmap(dpi=100)
    pix.save(os.path.join(outdir, "p%02d.png" % (i + 1)))
print("rendered", doc.page_count)
