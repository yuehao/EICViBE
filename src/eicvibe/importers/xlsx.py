"""Excel workbooks, read with the standard library: an .xlsx file is a zip of XML.

Design data come in workbooks (the EIN preinjector's survey, the parameter databases of its
subsystems). Only cell values are read, as the workbook last saved them (a formula's cached
result); formulas, formats and pictures are not.
"""

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def read_xlsx(path: str | Path) -> dict[str, list[list[str]]]:
    """Every sheet of a workbook, by name, as rows of cell texts (empty cells as ""), read with the
    standard library: an .xlsx file is a zip of XML."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t")))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        sheets = {}
        for sheet in ET.fromstring(z.read("xl/workbook.xml")).find("m:sheets", ns):
            target = targets[sheet.get(rel)].lstrip("/")
            target = target if target.startswith("xl/") else f"xl/{target}"
            rows = []
            for row in ET.fromstring(z.read(target)).find("m:sheetData", ns):
                cells: dict[int, str] = {}
                for c in row.findall("m:c", ns):
                    column = _column(c.get("r", ""))
                    v = c.find("m:v", ns)
                    if v is not None:
                        cells[column] = shared[int(v.text)] if c.get("t") == "s" else v.text or ""
                    elif (inline := c.find("m:is", ns)) is not None:
                        cells[column] = "".join(t.text or "" for t in inline.iter(f"{{{ns['m']}}}t"))
                rows.append([cells.get(k, "") for k in range(max(cells) + 1)] if cells else [])
            sheets[sheet.get("name")] = rows
    return sheets


def _column(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref)
    value = 0
    for ch in letters.group(0) if letters else "A":
        value = value * 26 + ord(ch) - 64
    return value - 1
