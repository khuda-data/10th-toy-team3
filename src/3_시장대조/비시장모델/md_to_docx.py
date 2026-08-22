"""Convert the project Markdown report to a portable Word (.docx) report.

No third-party package is required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
REPORT_MARKDOWN = ROOT / "docs" / "비시장_경마_우승확률_모델_중간보고서.md"
OUTPUT_DOCX = ROOT / "docs" / "비시장_경마_우승확률_모델_중간보고서.docx"
NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def run(text: str, style: str | None = None, bold: bool = False) -> str:
    properties = "<w:rPr><w:rFonts w:ascii=\"Malgun Gothic\" w:hAnsi=\"Malgun Gothic\" w:eastAsia=\"Malgun Gothic\"/>"
    if bold:
        properties += "<w:b/>"
    properties += "</w:rPr>"
    return f"<w:r>{properties}<w:t xml:space=\"preserve\">{escape(text)}</w:t></w:r>"


def paragraph(text: str = "", style: str | None = None, bold: bool = False) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}{run(text, bold=bold)}</w:p>"


def table(rows: list[list[str]]) -> str:
    cells = []
    for row_index, row in enumerate(rows):
        row_xml = []
        for value in row:
            shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
            row_xml.append(
                "<w:tc><w:tcPr><w:tcW w:w=\"1800\" w:type=\"dxa\"/>"
                f"{shade}</w:tcPr>{paragraph(value, bold=row_index == 0)}</w:tc>"
            )
        cells.append("<w:tr>" + "".join(row_xml) + "</w:tr>")
    return '<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="2"/><w:insideV w:val="single" w:sz="2"/></w:tblBorders></w:tblPr>' + "".join(cells) + "</w:tbl>"


def markdown_body(markdown: str) -> str:
    body: list[str] = []
    lines = markdown.splitlines()
    index = 0
    in_code = False
    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            in_code = not in_code
            index += 1
            continue
        if in_code:
            body.append(paragraph(line, "Code"))
        elif line.startswith("| ") and line.endswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                values = [value.strip() for value in lines[index].strip("|").split("|")]
                if not all(set(value) <= {"-", ":"} for value in values):
                    rows.append(values)
                index += 1
            body.append(table(rows))
            continue
        elif line.startswith("# "):
            body.append(paragraph(line[2:], "Title", bold=True))
        elif line.startswith("## "):
            body.append(paragraph(line[3:], "Heading1", bold=True))
        elif line.startswith("### "):
            body.append(paragraph(line[4:], "Heading2", bold=True))
        elif line.startswith("- "):
            body.append(paragraph("• " + line[2:]))
        elif line and line[0].isdigit() and ". " in line[:4]:
            body.append(paragraph(line))
        elif line.startswith("> "):
            body.append(paragraph(line[2:], "Quote"))
        elif line.strip():
            body.append(paragraph(line.replace("**", "")))
        else:
            body.append(paragraph())
        index += 1
    return "".join(body)


def main() -> None:
    source = REPORT_MARKDOWN
    target = OUTPUT_DOCX
    if not source.exists():
        raise FileNotFoundError(source)
    body = markdown_body(source.read_text(encoding="utf-8"))
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS}"><w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>'''
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{NS}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Malgun Gothic" w:hAnsi="Malgun Gothic" w:eastAsia="Malgun Gothic"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:sz w:val="34"/><w:b/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:rPr><w:sz w:val="28"/><w:b/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:rPr><w:sz w:val="24"/><w:b/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Quote"><w:name w:val="Quote"/><w:pPr><w:ind w:left="360"/></w:pPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>'''
    relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>'''
    document_relationships = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    core = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"><dc:title>비시장 경마 우승확률 모델 중간보고서</dc:title><dc:creator>Codex</dc:creator><dcterms:created>{now}</dcterms:created></cp:coreProperties>'''
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", document_relationships)
        archive.writestr("docProps/core.xml", core)
    print(target.resolve())


if __name__ == "__main__":
    main()
