# -*- coding: utf-8 -*-
"""Word 报告排版助手：样式、公式(OMML)、三线表、代码块、图表"""
import copy
import os
import re
import subprocess
import tempfile
import zipfile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from lxml import etree

PANDOC = r"D:\anaconda3\Library\bin\pandoc.exe"
M_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
BODY_FONT, HEAD_FONT, CODE_FONT = "宋体", "黑体", "Consolas"
GREY = "F2F2F2"


# ------------------------------------------------------------------ 基础字体
def set_run_font(run, name=BODY_FONT, size=12, bold=False, ascii_font=None):
    run.font.name = ascii_font or name              # 西文字体
    run.font.size = Pt(size)
    run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)              # 中文字体
    rfonts.set(qn("w:ascii"), ascii_font or name)
    rfonts.set(qn("w:hAnsi"), ascii_font or name)


def set_style_font(style, name, size, bold=False, ascii_font=None):
    style.font.name = ascii_font or name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), name)
    rfonts.set(qn("w:ascii"), ascii_font or name)
    rfonts.set(qn("w:hAnsi"), ascii_font or name)


# ------------------------------------------------------------------ 页面与样式
def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)      # A4
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(2.8)
    sec.right_margin = Cm(2.8)
    sec.different_first_page_header_footer = True               # 封面不要页码

    normal = doc.styles["Normal"]
    set_style_font(normal, BODY_FONT, 12)                       # 正文宋体小四
    pf = normal.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE        # 1.5 倍行距
    pf.space_before, pf.space_after = Pt(0), Pt(0)
    pf.first_line_indent = Pt(24)                                # 首行缩进 2 字符
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    for name, size, before, after in [("Heading 1", 15, 14, 8),
                                      ("Heading 2", 13, 10, 6),
                                      ("Heading 3", 12, 8, 4)]:
        st = doc.styles[name]
        set_style_font(st, HEAD_FONT, size, bold=True)           # 标题黑体
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.first_line_indent = Pt(0)
        st.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        st.paragraph_format.keep_with_next = True

    add_page_number_footer(doc)
    return doc


def add_page_number_footer(doc):
    """在页脚居中插入自动页码（封面不显示）"""
    sec = doc.sections[0]
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    for tag, attr in [("w:fldChar", {"w:fldCharType": "begin"}),
                      ("w:instrText", {"xml:space": "preserve"}),
                      ("w:fldChar", {"w:fldCharType": "end"})]:
        el = OxmlElement(tag)
        for k, v in attr.items():
            el.set(qn(k), v)
        if tag == "w:instrText":
            el.text = "PAGE"
        run._r.append(el)
    set_run_font(run, BODY_FONT, 10.5)


# ------------------------------------------------------------------ 段落
def h1(doc, text):
    return doc.add_paragraph(text, style="Heading 1")


def h2(doc, text):
    return doc.add_paragraph(text, style="Heading 2")


def h3(doc, text):
    return doc.add_paragraph(text, style="Heading 3")


def para(doc, text, indent=True, size=12, bold=False, spacing=1.5,
         align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=0):
    """普通正文段落；indent=False 时不缩进（用于列表式行）"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.first_line_indent = Pt(24) if indent else Pt(0)
    pf.line_spacing_rule = (WD_LINE_SPACING.ONE_POINT_FIVE if spacing == 1.5
                            else WD_LINE_SPACING.SINGLE)
    pf.alignment = align
    pf.space_after = Pt(space_after)
    # 支持 **粗体** 行内标记
    for i, seg in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            r = p.add_run(seg[2:-2])
            set_run_font(r, BODY_FONT, size, bold=True)
        else:
            r = p.add_run(seg)
            set_run_font(r, BODY_FONT, size, bold=bold)
    return p


def bullet(doc, marker, text, size=12):
    """带悬挂缩进的条目，如 (1) xxx / ① xxx"""
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.left_indent = Pt(24)
    pf.first_line_indent = Pt(0)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.alignment = WD_ALIGN_PARAGRAPH.LEFT     # 条目里英文/公式较多，左对齐避免拉伸
    pf.space_after = Pt(0)
    for i, seg in enumerate(re.split(r"(\*\*[^*]+\*\*)", marker + text)):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            r = p.add_run(seg[2:-2])
            set_run_font(r, BODY_FONT, size, bold=True)
        else:
            r = p.add_run(seg)
            set_run_font(r, BODY_FONT, size)
    return p


# ------------------------------------------------------------------ 公式
_omml_cache = {}


def latex_to_omml(latex):
    """用 pandoc 把 LaTeX 公式转成 Word 原生 OMML 公式（display 模式：行内独立、上下限完整）"""
    if latex in _omml_cache:
        return copy.deepcopy(_omml_cache[latex])
    tmp = tempfile.mkdtemp()
    md = os.path.join(tmp, "f.md")
    dx = os.path.join(tmp, "f.docx")
    with open(md, "w", encoding="utf-8") as fp:
        fp.write("$$" + latex + "$$\n")          # $$...$$ => display 公式
    subprocess.run([PANDOC, md, "-o", dx], capture_output=True, check=True)
    with zipfile.ZipFile(dx) as z:
        xml = z.read("word/document.xml")
    tree = etree.fromstring(xml)
    node = tree.find(".//" + M_NS + "oMathPara")   # 优先取 display 段落
    if node is None:
        node = tree.find(".//" + M_NS + "oMath")
    if node is None:
        raise RuntimeError("pandoc 未产生 OMML: " + latex)
    _omml_cache[latex] = copy.deepcopy(node)
    return copy.deepcopy(node)


def equation(doc, latex, number=None, first_cm=1.5, mid_cm=12.4, last_cm=1.5):
    """独立公式：三栏无边框表格，中间栏放 display 公式并居中，右栏放编号"""
    t = doc.add_table(rows=1, cols=3)
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    _fix_table_layout(t, [first_cm, mid_cm, last_cm])
    cells = t.rows[0].cells
    for c, w in zip(cells, [Cm(first_cm), Cm(mid_cm), Cm(last_cm)]):
        c.width = w
        _set_cell_border(c, top={"val": "nil"}, left={"val": "nil"},
                         bottom={"val": "nil"}, right={"val": "nil"})
        tcPr = c._tc.get_or_add_tcPr()
        mar = OxmlElement("w:tcMar")               # 收紧单元格边距
        for side in ("top", "left", "bottom", "right"):
            el = OxmlElement("w:" + side)
            el.set(qn("w:w"), "0")
            el.set(qn("w:type"), "dxa")
            mar.append(el)
        tcPr.append(mar)
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    p0 = cells[1].paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.first_line_indent = Pt(0)
    p0.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p0.paragraph_format.space_before = Pt(4)
    p0.paragraph_format.space_after = Pt(4)
    p0._p.append(latex_to_omml(latex))

    if number is not None:
        p2 = cells[2].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p2.paragraph_format.first_line_indent = Pt(0)
        p2.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p2.add_run("(%s)" % number)
        set_run_font(r, BODY_FONT, 12)
    return t


# ------------------------------------------------------------------ 代码块
def _display_em(line):
    """估算一行的显示宽度（中文/全角算 1 em，半角算 0.55 em），用于选字号"""
    return sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in line)


def code_block(doc, code, size=None, avail_pt=437.0):
    """浅灰底纹、Consolas 代码块（每行一段，段间距 0）。字号按最长行自适应，避免折行"""
    lines = code.strip("\n").split("\n")
    if size is None:
        max_em = max(_display_em(l) for l in lines) or 1.0
        size = min(10.5, avail_pt / max_em)
        size = max(8.0, int(size * 2) / 2.0)        # 向下取到 0.5 的倍数，最小 8pt
    paras = []
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.left_indent = Pt(0)
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
        pf.space_before = Pt(4 if i == 0 else 0)
        pf.space_after = Pt(4 if i == len(lines) - 1 else 0)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.keep_together = True
        # 浅灰底纹
        pPr = p._p.get_or_add_pPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), GREY)
        pPr.append(shd)
        r = p.add_run(line if line else " ")
        set_run_font(r, CODE_FONT, size, ascii_font=CODE_FONT)
        paras.append(p)
    return paras


# ------------------------------------------------------------------ 图表
def figure(doc, path, caption, width_cm=14.8):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(path, width=Cm(width_cm))
    cp = doc.add_paragraph()
    cp.paragraph_format.first_line_indent = Pt(0)
    cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_after = Pt(8)
    cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = cp.add_run(caption)
    set_run_font(r, BODY_FONT, 10.5, bold=True)
    return p


def _fix_table_layout(t, col_widths_cm):
    """强制固定列宽：tblW + tblLayout=fixed + gridCol 宽度，否则 Word 会按内容自动调宽"""
    tbl = t._tbl
    tblPr = tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        el = tblPr.find(qn(tag))
        if el is not None:
            tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"), str(int(sum(col_widths_cm) * 567)))
    tblW.set(qn("w:type"), "dxa")
    tblPr.append(tblW)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    for j, w in enumerate(col_widths_cm):
        t.columns[j].width = Cm(w)
        for row in t.rows:
            row.cells[j].width = Cm(w)


def _set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        if edge in kwargs:
            spec = kwargs[edge]
            tag = "w:" + edge
            el = borders.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                borders.append(el)
            for k, v in spec.items():
                el.set(qn("w:" + k), str(v))


def table(doc, header, rows, caption, col_widths=None, font_size=10.5,
          align_first_left=False, caption_above=True):
    """三线表：顶线/底线 1.5 磅，表头下 1 磅，无竖线"""
    if caption_above and caption:
        cp = doc.add_paragraph()
        cp.paragraph_format.first_line_indent = Pt(0)
        cp.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_before = Pt(8)
        cp.paragraph_format.space_after = Pt(3)
        cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        cp.paragraph_format.keep_with_next = True
        r = cp.add_run(caption)
        set_run_font(r, BODY_FONT, 10.5, bold=True)

    n_cols = len(header)
    t = doc.add_table(rows=1 + len(rows), cols=n_cols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    thick = {"val": "single", "sz": 12, "space": 0, "color": "000000"}
    thin = {"val": "single", "sz": 8, "space": 0, "color": "000000"}
    none = {"val": "nil"}

    def fill(cell, text, bold=False):
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        p.paragraph_format.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (align_first_left and not bold)
                                        else WD_ALIGN_PARAGRAPH.CENTER)
        r = p.add_run(str(text))
        set_run_font(r, BODY_FONT, font_size, bold=bold)

    for j, h in enumerate(header):
        fill(t.rows[0].cells[j], h, bold=True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            fill(t.rows[i + 1].cells[j], v)

    for i, row in enumerate(t.rows):
        for cell in row.cells:
            top = thick if i == 0 else (thin if i == 1 else none)
            bottom = thick if i == len(t.rows) - 1 else (thin if i == 1 else none)
            _set_cell_border(cell, top=top, bottom=bottom, left=none, right=none)

    # 行不跨页；除最后一行外都设 keep_with_next，尽量让整张表落在同一页
    for i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        if i == 0:
            hdr = OxmlElement("w:tblHeader")      # 跨页时重复表头
            trPr.append(hdr)
    for row in t.rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:
                p.paragraph_format.keep_with_next = True

    if col_widths:
        _fix_table_layout(t, col_widths)
    return t


def caption_below(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = p.add_run(text)
    set_run_font(r, BODY_FONT, 10.5, bold=True)
    return p


def page_break(doc):
    doc.add_page_break()


def blank(doc, n=1, size=12):
    for _ in range(n):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        r = p.add_run("")
        set_run_font(r, BODY_FONT, size)
