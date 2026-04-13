#!/usr/bin/env python3
"""
USB Forensic Analyzer — Lab 4b2
Professional GUI + PDF Report Generator
Replaces USB Historian with a modern Python DFIR toolkit.

Dependencies:
    pip install reportlab matplotlib pandas pillow regipy python-evtx rich
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import re
import csv
import sys
import io
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, white, black

try:
    from regipy.registry import RegistryHive
    REGIPY = True
except ImportError:
    REGIPY = False

try:
    from Evtx.Evtx import Evtx
    from Evtx.Views import evtx_file_xml_view
    EVTX = True
except ImportError:
    EVTX = False

# ─── Color palette ────────────────────────────────────────────────────────────
C_BG        = "#0D1117"
C_PANEL     = "#161B22"
C_BORDER    = "#21262D"
C_ACCENT    = "#00D4FF"
C_ACCENT2   = "#7C3AED"
C_DANGER    = "#F85149"
C_WARN      = "#F0A33A"
C_SUCCESS   = "#3FB950"
C_TEXT      = "#E6EDF3"
C_MUTED     = "#7D8590"
C_HEADER_BG = "#1C2128"

# ─── Demo data ─────────────────────────────────────────────────────────────────
DEMO_DEVICES = [
    {"source":"USBSTOR","device_class":"Disk&Ven_SanDisk&Prod_Ultra&Rev_1.00",
     "serial":"4C530001151218107130","friendly_name":"SanDisk Ultra USB Device",
     "hardware_id":"USBSTOR\\DiskSanDisk","vid":"0781","pid":"5581","revision":"1.00",
     "first_install":"2024-01-15 07:30:22","last_arrival":"2024-01-15 07:30:22","drive_letter":"E:","anomaly":"clean"},
    {"source":"USBSTOR","device_class":"Disk&Ven_Kingston&Prod_DataTraveler&Rev_PMAP",
     "serial":"60A44C413DF8E9A0A6A8","friendly_name":"Kingston DataTraveler 3.0",
     "hardware_id":"USBSTOR\\DiskKingston","vid":"0951","pid":"1666","revision":"PMAP",
     "first_install":"2024-01-15 09:12:44","last_arrival":"2024-01-15 09:12:44","drive_letter":"D:","anomaly":"clean"},
    {"source":"USBSTOR","device_class":"Disk&Ven_Generic&Prod_Flash_Disk&Rev_8.07",
     "serial":"058F63666436","friendly_name":"Generic Flash Disk USB Device",
     "hardware_id":"USBSTOR\\DiskGeneric","vid":"058F","pid":"6366","revision":"8.07",
     "first_install":"2024-01-15 10:47:03","last_arrival":"2024-01-15 10:47:03","drive_letter":"F:","anomaly":"clean"},
    {"source":"USB_ENUM","device_class":"VID_046D&PID_C52B","serial":"5&2CF82A4B&0&1",
     "friendly_name":"Logitech USB Receiver","hardware_id":"USB\\VID_046D&PID_C52B",
     "vid":"046D","pid":"C52B","revision":"","first_install":"2023-11-05 08:00:00",
     "last_arrival":"2024-01-14 17:00:00","drive_letter":"","anomaly":"clean"},
    {"source":"USBSTOR","device_class":"Disk&Ven_UNKNOWN&Prod_DEVICE&Rev_0000",
     "serial":"001","friendly_name":"","hardware_id":"USBSTOR\\DiskUNKNOWN",
     "vid":"DEAD","pid":"BEEF","revision":"0000",
     "first_install":"2024-01-15 14:32:11","last_arrival":"2024-01-15 14:32:11","drive_letter":"G:","anomaly":"unknown_vid, suspicious_serial"},
    {"source":"USBSTOR","device_class":"Disk&Ven_Transcend&Prod_JetFlash&Rev_8.07",
     "serial":"TS128GJF790","friendly_name":"Transcend JetFlash 790",
     "hardware_id":"USBSTOR\\DiskTranscend","vid":"8564","pid":"1000","revision":"8.07",
     "first_install":"2024-01-14 15:20:00","last_arrival":"2024-01-14 15:20:00","drive_letter":"H:","anomaly":"unknown_vid"},
]

KNOWN_VIDS = {"0781","0930","0951","04E8","0BC2","1058","058F","046D","045E","8564","0411"}

# ─── Registry / log parsers (from previous tool) ──────────────────────────────
def parse_usbstor(hive_path):
    if not REGIPY or not Path(hive_path).exists():
        return []
    results = []
    try:
        hive = RegistryHive(str(hive_path))
        usbstor_key = hive.get_key("\\SYSTEM\\CurrentControlSet\\Enum\\USBSTOR")
    except Exception:
        return []
    for device_class_key in usbstor_key.iter_subkeys():
        class_name = device_class_key.name
        vid = re.search(r"Ven_([^&]+)", class_name)
        pid = re.search(r"Prod_([^&]+)", class_name)
        rev = re.search(r"Rev_([^&]+)", class_name)
        for serial_key in device_class_key.iter_subkeys():
            def gv(n, d=""):
                try: return serial_key.get_value(n)
                except: return d
            ts = serial_key.header.last_modified
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, datetime) else str(ts)
            results.append({
                "source":"USBSTOR","device_class":class_name,
                "serial":serial_key.name.rstrip("&0"),
                "friendly_name":gv("FriendlyName"),"hardware_id":gv("HardwareID"),
                "vid":vid.group(1) if vid else "","pid":pid.group(1) if pid else "",
                "revision":rev.group(1) if rev else "","first_install":"",
                "last_arrival":ts_str,"drive_letter":"","anomaly":"",
            })
    return results

RE_SECTION_START = re.compile(r">>>  \[Device Install.*?- (USB\\[^\]]+)\]", re.IGNORECASE)
RE_TIMESTAMP     = re.compile(r">>>  Section start (\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})")

def parse_setupapi(log_path):
    if not Path(log_path).exists():
        return {}
    installs = {}
    current_device = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = RE_SECTION_START.search(line)
                if m: current_device = m.group(1); continue
                m2 = RE_TIMESTAMP.search(line)
                if m2 and current_device:
                    installs.setdefault(current_device, []).append(m2.group(1))
                    current_device = None
    except OSError:
        return {}
    return {k: {"first_install": sorted(v)[0], "last_install": sorted(v)[-1]}
            for k, v in installs.items()}

def flag_anomalies(devices):
    for dev in devices:
        flags = []
        vid = dev.get("vid", "").upper()
        if vid and vid not in KNOWN_VIDS:
            flags.append("unknown_vid")
        friendly = dev.get("friendly_name", "")
        known_vendors = {"Kingston","SanDisk","Samsung","Seagate","WD","Toshiba",
                         "Verbatim","Transcend","Lexar","Corsair","Microsoft","Logitech","Generic"}
        if friendly and not any(v.lower() in friendly.lower() for v in known_vendors):
            flags.append("unknown_vendor")
        serial = dev.get("serial", "")
        if serial and (len(serial) < 4 or serial == "0" * len(serial)):
            flags.append("suspicious_serial")
        if not dev.get("anomaly"):
            dev["anomaly"] = ", ".join(flags) if flags else "clean"
    return devices

def filter_by_window(devices, start_str, end_str):
    if not start_str and not end_str:
        return devices
    fmts = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"]
    def to_dt(s):
        for f in fmts:
            try: return datetime.strptime(s.replace("/","-").split(".")[0], f)
            except: pass
        return None
    dt_s = to_dt(start_str) if start_str else None
    dt_e = to_dt(end_str)   if end_str   else None
    matched = []
    for dev in devices:
        ts = dev.get("first_install") or dev.get("last_arrival") or ""
        dt = to_dt(ts)
        if dt is None: continue
        if dt_s and dt < dt_s: continue
        if dt_e and dt > dt_e: continue
        matched.append(dev)
    return matched


# ─── Chart generators ─────────────────────────────────────────────────────────
CHART_STYLE = {
    "bg":       "#0D1117",
    "panel":    "#161B22",
    "text":     "#E6EDF3",
    "muted":    "#7D8590",
    "accent":   "#00D4FF",
    "danger":   "#F85149",
    "warn":     "#F0A33A",
    "success":  "#3FB950",
    "purple":   "#7C3AED",
    "grid":     "#21262D",
}

def _apply_dark_style(fig, axes_list):
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    for ax in axes_list:
        ax.set_facecolor(CHART_STYLE["panel"])
        ax.tick_params(colors=CHART_STYLE["muted"], labelsize=8)
        ax.xaxis.label.set_color(CHART_STYLE["muted"])
        ax.yaxis.label.set_color(CHART_STYLE["muted"])
        ax.title.set_color(CHART_STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(CHART_STYLE["grid"])
        ax.grid(color=CHART_STYLE["grid"], linewidth=0.5, alpha=0.8)


def make_timeline_chart(devices):
    """Horizontal bar chart of device first-seen timestamps."""
    fmts = ["%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M","%Y/%m/%d %H:%M:%S"]
    def to_dt(s):
        for f in fmts:
            try: return datetime.strptime(s.replace("/","-").split(".")[0], f)
            except: pass
        return None

    items = []
    for d in devices:
        ts = d.get("first_install") or d.get("last_arrival") or ""
        dt = to_dt(ts)
        if dt:
            label = (d.get("friendly_name") or d.get("serial","?"))[:28]
            color = CHART_STYLE["danger"] if d.get("anomaly","clean") != "clean" else CHART_STYLE["accent"]
            items.append((dt, label, color))
    items.sort(key=lambda x: x[0])

    if not items:
        return None

    fig, ax = plt.subplots(figsize=(9, max(3, len(items) * 0.55 + 1)))
    _apply_dark_style(fig, [ax])

    dates  = [x[0] for x in items]
    labels = [x[1] for x in items]
    cols   = [x[2] for x in items]
    y_pos  = range(len(items))

    ax.scatter(dates, y_pos, c=cols, s=80, zorder=3, edgecolors="none")
    for i, (d, lbl, c) in enumerate(items):
        ax.annotate(lbl, (d, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=CHART_STYLE["text"])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    fig.autofmt_xdate(rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_title("Device connection timeline", fontsize=10, pad=8)
    ax.set_xlabel("Timestamp", fontsize=8)

    clean_p  = mpatches.Patch(color=CHART_STYLE["accent"], label="Clean")
    flagged_p = mpatches.Patch(color=CHART_STYLE["danger"], label="Flagged")
    ax.legend(handles=[clean_p, flagged_p], fontsize=7,
              facecolor=CHART_STYLE["panel"], edgecolor=CHART_STYLE["grid"],
              labelcolor=CHART_STYLE["text"])

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_STYLE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf


def make_anomaly_pie(devices):
    """Donut chart: clean vs flagged devices."""
    clean   = sum(1 for d in devices if d.get("anomaly","clean") == "clean")
    flagged = len(devices) - clean

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    _apply_dark_style(fig, [ax])

    if flagged == 0:
        vals   = [1]
        clrs   = [CHART_STYLE["success"]]
        lbls   = ["All clean"]
    else:
        vals   = [clean, flagged]
        clrs   = [CHART_STYLE["success"], CHART_STYLE["danger"]]
        lbls   = [f"Clean ({clean})", f"Flagged ({flagged})"]

    wedges, texts, autotexts = ax.pie(
        vals, colors=clrs, labels=lbls, autopct="%1.0f%%",
        startangle=90, wedgeprops={"width": 0.55, "edgecolor": CHART_STYLE["bg"], "linewidth": 2},
        pctdistance=0.75,
    )
    for t in texts:     t.set_color(CHART_STYLE["text"]); t.set_fontsize(8)
    for t in autotexts: t.set_color(CHART_STYLE["bg"]);  t.set_fontsize(8); t.set_fontweight("bold")

    ax.set_title("Anomaly status", fontsize=10, pad=6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_STYLE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf


def make_vendor_bar(devices):
    """Bar chart of top device vendors/VIDs."""
    from collections import Counter
    vendor_counts = Counter()
    for d in devices:
        name = d.get("friendly_name", "")
        vid  = d.get("vid", "UNKNOWN")
        label = name.split()[0] if name else f"VID_{vid}"
        vendor_counts[label] += 1

    top = vendor_counts.most_common(8)
    if not top:
        return None

    labels = [x[0] for x in top]
    counts = [x[1] for x in top]
    bar_colors = [CHART_STYLE["danger"] if c.startswith("VID_DEAD") or c == "VID_" else
                  CHART_STYLE["accent"] for c in labels]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    _apply_dark_style(fig, [ax])

    bars = ax.barh(labels[::-1], counts[::-1], color=bar_colors[::-1],
                   edgecolor="none", height=0.6)
    for bar, val in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=8, color=CHART_STYLE["text"])

    ax.set_xlabel("Count", fontsize=8)
    ax.set_title("Devices by vendor", fontsize=10, pad=8)
    ax.set_xlim(0, max(counts) + 1.5)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_STYLE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf


def make_hourly_histogram(devices):
    """Histogram of connection events by hour of day."""
    fmts = ["%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M"]
    hours = []
    for d in devices:
        ts = d.get("first_install") or d.get("last_arrival") or ""
        for f in fmts:
            try:
                dt = datetime.strptime(ts.split(".")[0], f)
                hours.append(dt.hour)
                break
            except: pass

    if not hours:
        return None

    fig, ax = plt.subplots(figsize=(6, 2.8))
    _apply_dark_style(fig, [ax])

    ax.bar(range(24), [hours.count(h) for h in range(24)],
           color=CHART_STYLE["purple"], edgecolor="none", width=0.75)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Events", fontsize=8)
    ax.set_title("Connection events by hour", fontsize=10, pad=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=CHART_STYLE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf


# ─── PDF Report ───────────────────────────────────────────────────────────────
REPORT_COLORS = {
    "bg_dark":  HexColor("#0D1117"),
    "bg_panel": HexColor("#161B22"),
    "accent":   HexColor("#00D4FF"),
    "accent2":  HexColor("#7C3AED"),
    "danger":   HexColor("#F85149"),
    "warn":     HexColor("#F0A33A"),
    "success":  HexColor("#3FB950"),
    "text":     HexColor("#E6EDF3"),
    "muted":    HexColor("#7D8590"),
    "border":   HexColor("#21262D"),
    "header":   HexColor("#1C2128"),
}

def _build_styles():
    base = getSampleStyleSheet()
    custom = {}

    def add(name, parent="Normal", **kw):
        s = ParagraphStyle(name, parent=base[parent], **kw)
        custom[name] = s

    add("Cover_Title", fontSize=32, textColor=white,
        fontName="Helvetica-Bold", leading=38, alignment=TA_LEFT)
    add("Cover_Sub",   fontSize=14, textColor=REPORT_COLORS["muted"],
        fontName="Helvetica", leading=20, alignment=TA_LEFT)
    add("Cover_Meta",  fontSize=9,  textColor=REPORT_COLORS["accent"],
        fontName="Helvetica", leading=14, alignment=TA_LEFT)
    add("Section_H1",  fontSize=16, textColor=REPORT_COLORS["accent"],
        fontName="Helvetica-Bold", leading=22, spaceBefore=18, spaceAfter=6)
    add("Section_H2",  fontSize=11, textColor=REPORT_COLORS["text"],
        fontName="Helvetica-Bold", leading=16, spaceBefore=10, spaceAfter=4)
    add("Body",        fontSize=9,  textColor=REPORT_COLORS["text"],
        fontName="Helvetica", leading=14, spaceAfter=4)
    add("Body_Mono",   fontSize=8,  textColor=REPORT_COLORS["accent"],
        fontName="Courier", leading=13)
    add("Danger",      fontSize=9,  textColor=REPORT_COLORS["danger"],
        fontName="Helvetica-Bold", leading=14)
    add("Warn",        fontSize=9,  textColor=REPORT_COLORS["warn"],
        fontName="Helvetica", leading=14)
    add("Success",     fontSize=9,  textColor=REPORT_COLORS["success"],
        fontName="Helvetica", leading=14)
    add("Caption",     fontSize=8,  textColor=REPORT_COLORS["muted"],
        fontName="Helvetica", leading=11, alignment=TA_CENTER, spaceAfter=8)
    add("Table_Header",fontSize=8, textColor=white,
        fontName="Helvetica-Bold", leading=11, alignment=TA_CENTER)
    add("Table_Cell",  fontSize=7.5,textColor=REPORT_COLORS["text"],
        fontName="Helvetica", leading=10)
    add("Table_Cell_Mono", fontSize=7, textColor=REPORT_COLORS["accent"],
        fontName="Courier", leading=10)
    add("Flagged_Cell",fontSize=7.5,textColor=REPORT_COLORS["danger"],
        fontName="Helvetica-Bold", leading=10)
    return custom


class DarkBackground(Flowable):
    """Full-width dark background stripe."""
    def __init__(self, height, color=None, radius=4):
        super().__init__()
        self.height = height
        self.color  = color or REPORT_COLORS["bg_panel"]
        self.radius = radius
        self.width  = 0

    def wrap(self, aW, aH):
        self.width = aW
        return aW, self.height

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height, self.radius, fill=1, stroke=0)


class AccentLine(Flowable):
    def __init__(self, color=None, thickness=2):
        super().__init__()
        self.color     = color or REPORT_COLORS["accent"]
        self.thickness = thickness
        self.width     = 0

    def wrap(self, aW, aH):
        self.width = aW
        return aW, self.thickness + 4

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.rect(0, 2, self.width, self.thickness, fill=1, stroke=0)


def _page_template(c, doc):
    """Background and footer on every page."""
    w, h = A4
    c.saveState()
    # Dark background
    c.setFillColor(REPORT_COLORS["bg_dark"])
    c.rect(0, 0, w, h, fill=1, stroke=0)
    # Left accent bar
    c.setFillColor(REPORT_COLORS["accent2"])
    c.rect(0, 0, 4, h, fill=1, stroke=0)
    # Footer
    c.setFillColor(REPORT_COLORS["bg_panel"])
    c.rect(0, 0, w, 22*mm, fill=1, stroke=0)
    c.setFillColor(REPORT_COLORS["accent"])
    c.rect(0, 22*mm, w, 0.5, fill=1, stroke=0)
    c.setFont("Helvetica", 7)
    c.setFillColor(REPORT_COLORS["muted"])
    c.drawString(18*mm, 8*mm, "USB Forensic Analyzer — Lab 4b2 | CONFIDENTIAL")
    c.drawRightString(w - 18*mm, 8*mm, f"Page {doc.page}")
    c.restoreState()


def _buf_to_rl_image(buf, width_mm, height_mm=None):
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(buf.read())
    tmp.close()
    if height_mm:
        return RLImage(tmp.name, width=width_mm*mm, height=height_mm*mm)
    return RLImage(tmp.name, width=width_mm*mm)


def generate_pdf_report(devices, output_path, analyst_name="", case_id="",
                        breach_start="", breach_end="", filtered_devices=None):
    styles = _build_styles()
    S = styles

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=22*mm, bottomMargin=28*mm,
        title="USB Forensic Analysis Report",
        author=analyst_name or "USB Forensic Analyzer",
    )

    story = []
    W = A4[0] - 36*mm  # usable width

    # ── Cover page ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 30*mm))
    story.append(AccentLine(color=REPORT_COLORS["accent"], thickness=3))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("USB FORENSIC", S["Cover_Title"]))
    story.append(Paragraph("ANALYSIS REPORT", S["Cover_Title"]))
    story.append(Spacer(1, 4*mm))
    story.append(AccentLine(color=REPORT_COLORS["accent2"], thickness=1))
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Digital Forensics &amp; Incident Response", S["Cover_Sub"]))
    story.append(Spacer(1, 8*mm))

    meta_rows = [
        ("Case ID",       case_id or "LAB-4B2"),
        ("Analyst",       analyst_name or "N/A"),
        ("Generated",     datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Classification","CONFIDENTIAL"),
    ]
    if breach_start or breach_end:
        meta_rows.insert(2, ("Breach window", f"{breach_start} → {breach_end}"))

    meta_table_data = [[Paragraph(f"<b>{k}</b>", S["Cover_Meta"]),
                        Paragraph(v, S["Cover_Meta"])] for k, v in meta_rows]
    mt = Table(meta_table_data, colWidths=[40*mm, W - 40*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), REPORT_COLORS["bg_panel"]),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [REPORT_COLORS["bg_panel"], REPORT_COLORS["header"]]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(mt)

    # Summary stats boxes
    story.append(Spacer(1, 16*mm))
    total   = len(devices)
    flagged = sum(1 for d in devices if d.get("anomaly","clean") != "clean")
    clean   = total - flagged
    in_win  = len(filtered_devices) if filtered_devices else 0

    stat_items = [
        (str(total),   "Total devices",   REPORT_COLORS["accent"]),
        (str(clean),   "Clean",           REPORT_COLORS["success"]),
        (str(flagged), "Flagged",         REPORT_COLORS["danger"]),
        (str(in_win),  "In breach window",REPORT_COLORS["warn"]),
    ]
    stat_cells = []
    for val, label, col in stat_items:
        cell_content = [
            Paragraph(f'<font color="#{col.hexval()[2:]}"><b>{val}</b></font>',
                      ParagraphStyle("sv", fontSize=22, fontName="Helvetica-Bold",
                                     textColor=col, alignment=TA_CENTER, leading=26)),
            Paragraph(label, ParagraphStyle("sl", fontSize=8, fontName="Helvetica",
                                            textColor=REPORT_COLORS["muted"],
                                            alignment=TA_CENTER, leading=11)),
        ]
        stat_cells.append(cell_content)

    col_w = W / 4
    stat_table = Table([stat_cells], colWidths=[col_w]*4, rowHeights=[22*mm])
    stat_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), REPORT_COLORS["bg_panel"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LINEAFTER",     (0,0), (2,-1), 0.5, REPORT_COLORS["border"]),
        ("ROUNDEDCORNERS",[4]),
    ]))
    story.append(stat_table)
    story.append(PageBreak())

    # ── Section 1: Executive Summary ──────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", S["Section_H1"]))
    story.append(AccentLine(color=REPORT_COLORS["border"], thickness=0.5))
    story.append(Spacer(1, 4*mm))

    breach_line = (f" A breach window was applied from <b>{breach_start}</b> to "
                   f"<b>{breach_end}</b>, narrowing the scope to "
                   f"<b>{in_win}</b> device(s) of interest."
                   if (breach_start or breach_end) else "")

    story.append(Paragraph(
        f"This report presents the findings of a USB artifact forensic analysis "
        f"performed on the target Windows system. A total of <b>{total}</b> USB device "
        f"records were identified across registry hives and installation logs. "
        f"Of these, <b>{clean}</b> device(s) were classified as clean and "
        f"<b>{flagged}</b> device(s) were flagged for further investigation based on "
        f"unknown vendor identifiers, suspicious serial numbers, or unrecognised "
        f"hardware IDs.{breach_line}", S["Body"]))
    story.append(Spacer(1, 3*mm))

    if flagged:
        story.append(Paragraph(
            f"&#9888;  {flagged} flagged device(s) require immediate investigator attention. "
            "Details are provided in Section 4.", S["Danger"]))

    # ── Section 2: Methodology ────────────────────────────────────────────────
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("2. Methodology &amp; Artifact Sources", S["Section_H1"]))
    story.append(AccentLine(color=REPORT_COLORS["border"], thickness=0.5))
    story.append(Spacer(1, 3*mm))

    methods = [
        ("USBSTOR Registry", "HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USBSTOR",
         "Storage device identity, serial numbers, VID/PID, last-write timestamps."),
        ("USB\\Enum Registry", "HKLM\\SYSTEM\\CurrentControlSet\\Enum\\USB",
         "All USB devices including hubs, HID, and non-storage peripherals."),
        ("SetupAPI Log", "C:\\Windows\\inf\\setupapi.dev.log",
         "Device installation timestamps; provides first-connection evidence."),
        ("Event Logs (EVTX)", "DriverFrameworks-UserMode Operational",
         "Arrival (Event 2003) and removal (Event 2100/2101) events."),
    ]
    for src, path, desc in methods:
        story.append(Paragraph(f"<b>{src}</b>", S["Section_H2"]))
        story.append(Paragraph(f'Path: <font color="#{REPORT_COLORS["accent"].hexval()[2:]}">{path}</font>', S["Body_Mono"]))
        story.append(Paragraph(desc, S["Body"]))

    # ── Section 3: Visual Analysis ────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Visual Analysis", S["Section_H1"]))
    story.append(AccentLine(color=REPORT_COLORS["border"], thickness=0.5))
    story.append(Spacer(1, 4*mm))

    # Timeline chart (full width)
    tl_buf = make_timeline_chart(devices)
    if tl_buf:
        img = _buf_to_rl_image(tl_buf, W/mm - 4, None)
        story.append(img)
        story.append(Paragraph("Figure 1 — Device connection timeline. Red markers indicate flagged devices.", S["Caption"]))

    story.append(Spacer(1, 4*mm))

    # Pie + Vendor bar side by side
    pie_buf = make_anomaly_pie(devices)
    bar_buf = make_vendor_bar(devices)

    if pie_buf and bar_buf:
        pie_img = _buf_to_rl_image(pie_buf, 78, 66)
        bar_img = _buf_to_rl_image(bar_buf, W/mm - 78 - 6, 66)
        side_table = Table([[pie_img, bar_img]],
                           colWidths=[80*mm, (W - 80*mm)])
        side_table.setStyle(TableStyle([
            ("ALIGN",  (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("LEFTPADDING",  (0,0), (-1,-1), 0),
            ("RIGHTPADDING", (0,0), (-1,-1), 0),
        ]))
        story.append(side_table)
        story.append(Paragraph(
            "Figure 2 — Anomaly distribution (left) and device vendor breakdown (right).",
            S["Caption"]))

    # Hourly histogram
    hour_buf = make_hourly_histogram(devices)
    if hour_buf:
        story.append(Spacer(1, 3*mm))
        img = _buf_to_rl_image(hour_buf, W/mm - 4, None)
        story.append(img)
        story.append(Paragraph(
            "Figure 3 — Connection events by hour of day. Unusual off-hours activity may indicate unauthorised access.",
            S["Caption"]))

    # ── Section 4: Device Inventory ───────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Device Inventory", S["Section_H1"]))
    story.append(AccentLine(color=REPORT_COLORS["border"], thickness=0.5))
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph(
        f"Complete listing of all {total} USB device(s) identified. "
        "Flagged entries are highlighted in red.", S["Body"]))
    story.append(Spacer(1, 3*mm))

    # Table header
    hdrs = ["#", "Serial / ID", "Friendly name", "VID/PID", "First seen", "Drive", "Status"]
    col_ws = [8*mm, 42*mm, 46*mm, 18*mm, 32*mm, 12*mm, 18*mm]
    table_data = [[Paragraph(h, S["Table_Header"]) for h in hdrs]]

    for i, dev in enumerate(devices, 1):
        is_flag = dev.get("anomaly", "clean") != "clean"
        cell_style = S["Flagged_Cell"] if is_flag else S["Table_Cell"]
        serial_style = S["Flagged_Cell"] if is_flag else S["Table_Cell_Mono"]

        status_text = dev.get("anomaly", "clean")
        if is_flag:
            status_para = Paragraph(f"&#9888; {status_text[:18]}", S["Flagged_Cell"])
        else:
            status_para = Paragraph("&#10003; clean", S["Success"])

        table_data.append([
            Paragraph(str(i), S["Table_Cell"]),
            Paragraph((dev.get("serial") or "")[:34], serial_style),
            Paragraph((dev.get("friendly_name") or dev.get("device_class",""))[:40], cell_style),
            Paragraph(f"{dev.get('vid','')}/{dev.get('pid','')}", S["Table_Cell_Mono"]),
            Paragraph((dev.get("first_install") or dev.get("last_arrival",""))[:19], S["Table_Cell"]),
            Paragraph(dev.get("drive_letter",""), S["Table_Cell"]),
            status_para,
        ])

    inv_table = Table(table_data, colWidths=col_ws, repeatRows=1)
    row_count  = len(table_data)
    style_cmds = [
        ("BACKGROUND",    (0,0), (-1,0),  REPORT_COLORS["accent2"]),
        ("BACKGROUND",    (0,1), (-1,-1), REPORT_COLORS["bg_panel"]),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [REPORT_COLORS["bg_panel"], REPORT_COLORS["header"]]),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("LINEBELOW",     (0,0), (-1,0),  0.5, REPORT_COLORS["accent"]),
        ("LINEBELOW",     (0,1), (-1,-2), 0.3, REPORT_COLORS["border"]),
        ("ROUNDEDCORNERS",[4]),
    ]
    # Highlight flagged rows
    for i, dev in enumerate(devices, 1):
        if dev.get("anomaly","clean") != "clean":
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), HexColor("#2D1B1B")))
            style_cmds.append(("LINEBELOW",  (0,i), (-1,i), 0.5, REPORT_COLORS["danger"]))

    inv_table.setStyle(TableStyle(style_cmds))
    story.append(inv_table)

    # ── Section 5: Flagged Devices Detail ─────────────────────────────────────
    flagged_devs = [d for d in devices if d.get("anomaly","clean") != "clean"]
    if flagged_devs:
        story.append(PageBreak())
        story.append(Paragraph("5. Flagged Devices — Detail", S["Section_H1"]))
        story.append(AccentLine(color=REPORT_COLORS["danger"], thickness=1))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            "The following devices were flagged by the anomaly detection engine. "
            "Each should be manually reviewed and correlated with user activity logs, "
            "file access events, and DLP controls.", S["Body"]))
        story.append(Spacer(1, 4*mm))

        for idx, dev in enumerate(flagged_devs, 1):
            story.append(Paragraph(f"&#9888;  Flagged Device #{idx}", S["Danger"]))
            detail_rows = [
                ("Serial / ID",    dev.get("serial","")),
                ("Friendly name",  dev.get("friendly_name","(none)")),
                ("VID / PID",      f"{dev.get('vid','')} / {dev.get('pid','')}"),
                ("Device class",   dev.get("device_class","")),
                ("Hardware ID",    dev.get("hardware_id","")),
                ("First seen",     dev.get("first_install") or dev.get("last_arrival","")),
                ("Drive letter",   dev.get("drive_letter","(unmapped)")),
                ("Anomaly flags",  dev.get("anomaly","")),
            ]
            d_data = [[Paragraph(f"<b>{k}</b>", S["Body"]),
                       Paragraph(v, S["Body_Mono"])] for k, v in detail_rows]
            dt = Table(d_data, colWidths=[38*mm, W - 38*mm])
            dt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), HexColor("#1A0D0D")),
                ("LINEAFTER",     (0,0), (0,-1),  0.5, REPORT_COLORS["danger"]),
                ("TOPPADDING",    (0,0), (-1,-1), 4),
                ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING",   (0,0), (-1,-1), 6),
                ("RIGHTPADDING",  (0,0), (-1,-1), 6),
                ("ROWBACKGROUNDS",(0,0), (-1,-1), [HexColor("#1A0D0D"), HexColor("#1F1010")]),
                ("ROUNDEDCORNERS",[4]),
            ]))
            story.append(dt)
            story.append(Spacer(1, 6*mm))

    # ── Section 6: Breach Window Analysis ─────────────────────────────────────
    if filtered_devices and (breach_start or breach_end):
        story.append(PageBreak())
        n_sec = 6 if flagged_devs else 5
        story.append(Paragraph(f"{n_sec}. Breach Window Analysis", S["Section_H1"]))
        story.append(AccentLine(color=REPORT_COLORS["warn"], thickness=1))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(
            f"Filtering all device records to the breach window "
            f"<b>{breach_start}</b> &#8594; <b>{breach_end}</b> "
            f"returned <b>{len(filtered_devices)}</b> matching device(s).", S["Body"]))
        story.append(Spacer(1, 4*mm))

        if filtered_devices:
            bw_hdrs = ["Serial", "Friendly name", "VID/PID", "First seen", "Status"]
            bw_cws  = [44*mm, 52*mm, 20*mm, 36*mm, 24*mm]
            bw_data = [[Paragraph(h, S["Table_Header"]) for h in bw_hdrs]]
            for dev in filtered_devices:
                is_f = dev.get("anomaly","clean") != "clean"
                bw_data.append([
                    Paragraph((dev.get("serial") or "")[:36],
                               S["Flagged_Cell"] if is_f else S["Table_Cell_Mono"]),
                    Paragraph((dev.get("friendly_name",""))[:44], S["Table_Cell"]),
                    Paragraph(f"{dev.get('vid','')}/{dev.get('pid','')}", S["Table_Cell_Mono"]),
                    Paragraph((dev.get("first_install") or dev.get("last_arrival",""))[:19], S["Table_Cell"]),
                    Paragraph("FLAGGED" if is_f else "clean",
                               S["Flagged_Cell"] if is_f else S["Success"]),
                ])
            bw_table = Table(bw_data, colWidths=bw_cws, repeatRows=1)
            bw_table.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0),  REPORT_COLORS["warn"]),
                ("BACKGROUND",    (0,1), (-1,-1), REPORT_COLORS["bg_panel"]),
                ("ROWBACKGROUNDS",(0,1), (-1,-1), [REPORT_COLORS["bg_panel"], REPORT_COLORS["header"]]),
                ("TOPPADDING",    (0,0), (-1,-1), 4), ("BOTTOMPADDING",(0,0),(-1,-1),4),
                ("LEFTPADDING",   (0,0), (-1,-1), 4), ("RIGHTPADDING", (0,0),(-1,-1),4),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("LINEBELOW",     (0,0), (-1,0),  0.5, REPORT_COLORS["warn"]),
                ("LINEBELOW",     (0,1), (-1,-2), 0.3, REPORT_COLORS["border"]),
                ("ROUNDEDCORNERS",[4]),
            ]))
            story.append(bw_table)

    # ── Section: Recommendations ──────────────────────────────────────────────
    story.append(PageBreak())
    last_sec = (6 if flagged_devs else 5) + (1 if (filtered_devices and (breach_start or breach_end)) else 0)
    story.append(Paragraph(f"{last_sec}. Recommendations", S["Section_H1"]))
    story.append(AccentLine(color=REPORT_COLORS["border"], thickness=0.5))
    story.append(Spacer(1, 4*mm))

    recs = [
        ("Immediate", f"Isolate and image the system if {flagged} flagged device(s) cannot be accounted for."),
        ("Short-term", "Cross-reference USB serial numbers against approved device inventory (MDM/DLP)."),
        ("Short-term", "Review file access logs (MFT, $UsnJrnl) for the drive letters mapped during the breach window."),
        ("Short-term", "Correlate with Active Directory login events to identify the logged-on user."),
        ("Preventive", "Deploy USB device control policies (e.g., Windows Defender Device Control) to block unknown VIDs."),
        ("Preventive", "Enable audit logging for removable storage: Security Event ID 6416, 4663."),
    ]
    for priority, text in recs:
        col = {"Immediate": REPORT_COLORS["danger"],
               "Short-term": REPORT_COLORS["warn"],
               "Preventive": REPORT_COLORS["accent"]}.get(priority, REPORT_COLORS["muted"])
        story.append(Paragraph(
            f'<font color="#{col.hexval()[2:]}"><b>[{priority}]</b></font>  {text}', S["Body"]))
        story.append(Spacer(1, 1*mm))

    # Build
    doc.build(story, onFirstPage=_page_template, onLaterPages=_page_template)
    # Cleanup tmp image files
    return output_path


# ─── GUI ──────────────────────────────────────────────────────────────────────
class USBForensicApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("USB Forensic Analyzer — Lab 4b2")
        self.geometry("1200x780")
        self.minsize(1000, 650)
        self.configure(bg=C_BG)
        self.devices = []
        self._build_ui()
        self._load_demo()

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._style_ttk()

        # ── Top bar ──
        topbar = tk.Frame(self, bg=C_HEADER_BG, height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)
        tk.Label(topbar, text="⬡  USB FORENSIC ANALYZER", bg=C_HEADER_BG,
                 fg=C_ACCENT, font=("Courier New", 13, "bold")).pack(side="left", padx=18, pady=14)
        tk.Label(topbar, text="Lab 4b2  •  DFIR Toolkit", bg=C_HEADER_BG,
                 fg=C_MUTED, font=("Courier New", 9)).pack(side="left")

        # Version badge
        badge = tk.Label(topbar, text=" v2.0 ", bg=C_ACCENT2, fg="white",
                         font=("Courier New", 8, "bold"), padx=6, pady=3)
        badge.pack(side="right", padx=18)

        # ── Main split ──
        paned = tk.PanedWindow(self, orient="horizontal", bg=C_BG,
                               sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=0, pady=0)

        left = tk.Frame(paned, bg=C_PANEL, width=310)
        left.pack_propagate(False)
        paned.add(left, minsize=260)

        right = tk.Frame(paned, bg=C_BG)
        paned.add(right, minsize=600)

        self._build_sidebar(left)
        self._build_main(right)

    def _style_ttk(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
                         background=C_PANEL, foreground=C_TEXT,
                         fieldbackground=C_PANEL, rowheight=26,
                         font=("Courier New", 9))
        style.configure("Treeview.Heading",
                         background=C_HEADER_BG, foreground=C_ACCENT,
                         font=("Courier New", 9, "bold"), relief="flat")
        style.map("Treeview",
                  background=[("selected", C_ACCENT2)],
                  foreground=[("selected", "white")])
        style.configure("TScrollbar", background=C_BORDER,
                        troughcolor=C_PANEL, arrowcolor=C_MUTED)
        style.configure("TNotebook", background=C_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=C_PANEL, foreground=C_MUTED,
                        padding=[12, 6], font=("Courier New", 9))
        style.map("TNotebook.Tab",
                  background=[("selected", C_HEADER_BG)],
                  foreground=[("selected", C_ACCENT)])

    def _section_label(self, parent, text):
        f = tk.Frame(parent, bg=C_PANEL)
        f.pack(fill="x", padx=12, pady=(12, 2))
        tk.Label(f, text=text, bg=C_PANEL, fg=C_ACCENT,
                 font=("Courier New", 8, "bold")).pack(side="left")
        tk.Frame(f, bg=C_BORDER, height=1).pack(fill="x", side="left",
                                                  expand=True, padx=(6, 0))

    def _labeled_entry(self, parent, label, var, width=28):
        f = tk.Frame(parent, bg=C_PANEL)
        f.pack(fill="x", padx=12, pady=2)
        tk.Label(f, text=label, bg=C_PANEL, fg=C_MUTED,
                 font=("Courier New", 8), width=14, anchor="w").pack(side="left")
        e = tk.Entry(f, textvariable=var, bg=C_BORDER, fg=C_TEXT,
                     insertbackground=C_ACCENT, relief="flat",
                     font=("Courier New", 9), width=width)
        e.pack(side="left", fill="x", expand=True, padx=(4, 0))
        return e

    def _browse_btn(self, parent, var, title="Select file"):
        def pick():
            path = filedialog.askopenfilename(title=title)
            if path: var.set(path)
        b = tk.Button(parent, text="…", bg=C_BORDER, fg=C_ACCENT,
                      relief="flat", font=("Courier New", 9, "bold"),
                      cursor="hand2", command=pick, padx=6)
        b.pack(side="right", padx=(2, 12), pady=2)
        return b

    def _build_sidebar(self, parent):
        # Scrollable sidebar
        canvas = tk.Canvas(parent, bg=C_PANEL, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C_PANEL)
        cwin = canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(cwin, width=e.width))

        # ── Input sources ──
        self._section_label(inner, "ARTIFACT SOURCES")

        self.v_system   = tk.StringVar()
        self.v_setupapi = tk.StringVar()
        self.v_evtx     = tk.StringVar()

        for label, var, hint in [
            ("SYSTEM hive",    self.v_system,   "Select SYSTEM registry hive"),
            ("SetupAPI log",   self.v_setupapi, "Select setupapi.dev.log"),
            ("EVTX log",       self.v_evtx,     "Select .evtx event log"),
        ]:
            row = tk.Frame(inner, bg=C_PANEL)
            row.pack(fill="x", padx=12, pady=2)
            tk.Label(row, text=label, bg=C_PANEL, fg=C_MUTED,
                     font=("Courier New", 8), anchor="w").pack(fill="x")
            ef = tk.Frame(row, bg=C_PANEL)
            ef.pack(fill="x")
            e = tk.Entry(ef, textvariable=var, bg=C_BORDER, fg=C_TEXT,
                         insertbackground=C_ACCENT, relief="flat",
                         font=("Courier New", 8))
            e.pack(side="left", fill="x", expand=True)
            tk.Button(ef, text="…", bg=C_ACCENT2, fg="white",
                      relief="flat", font=("Courier New", 9, "bold"),
                      cursor="hand2",
                      command=lambda v=var, h=hint: v.set(
                          filedialog.askopenfilename(title=h) or v.get()),
                      padx=6).pack(side="right", padx=(2,0))

        # ── Case info ──
        self._section_label(inner, "CASE INFORMATION")
        self.v_analyst  = tk.StringVar(value="Analyst")
        self.v_case_id  = tk.StringVar(value="CASE-001")
        self._labeled_entry(inner, "Analyst name", self.v_analyst, 18)
        self._labeled_entry(inner, "Case ID",      self.v_case_id, 18)

        # ── Breach window ──
        self._section_label(inner, "BREACH WINDOW FILTER")
        self.v_start = tk.StringVar(value="2024-01-15 09:00")
        self.v_end   = tk.StringVar(value="2024-01-15 17:00")
        self._labeled_entry(inner, "Start (YYYY-MM-DD HH:MM)", self.v_start, 18)
        self._labeled_entry(inner, "End   (YYYY-MM-DD HH:MM)", self.v_end,   18)

        # ── Actions ──
        self._section_label(inner, "ACTIONS")

        for text, cmd, bg, fg in [
            ("▶  Load Demo Data",   self._load_demo,         C_ACCENT2, "white"),
            ("⬡  Run Analysis",     self._run_analysis,      C_ACCENT,  C_BG),
            ("⬇  Export CSV",       self._export_csv,        C_BORDER,  C_TEXT),
            ("⎙  Generate PDF Report", self._generate_report, C_DANGER,  "white"),
        ]:
            tk.Button(inner, text=text, bg=bg, fg=fg, relief="flat",
                      font=("Courier New", 9, "bold"), cursor="hand2",
                      command=cmd, pady=9, padx=8).pack(fill="x", padx=12, pady=3)

        # ── Status ──
        self._section_label(inner, "STATUS")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(inner, textvariable=self.status_var, bg=C_PANEL, fg=C_SUCCESS,
                 font=("Courier New", 8), wraplength=260, justify="left").pack(
                     fill="x", padx=12, pady=4)

        # ── Stats panel ──
        self._section_label(inner, "SUMMARY")
        self.lbl_total   = self._stat_row(inner, "Total devices",    "0", C_ACCENT)
        self.lbl_clean   = self._stat_row(inner, "Clean",            "0", C_SUCCESS)
        self.lbl_flagged = self._stat_row(inner, "Flagged",          "0", C_DANGER)
        self.lbl_window  = self._stat_row(inner, "In breach window", "0", C_WARN)

    def _stat_row(self, parent, label, value, color):
        f = tk.Frame(parent, bg=C_PANEL)
        f.pack(fill="x", padx=12, pady=1)
        tk.Label(f, text=label, bg=C_PANEL, fg=C_MUTED,
                 font=("Courier New", 8)).pack(side="left")
        lbl = tk.Label(f, text=value, bg=C_PANEL, fg=color,
                       font=("Courier New", 10, "bold"))
        lbl.pack(side="right")
        return lbl

    def _build_main(self, parent):
        # Tab notebook
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        # Tab 1: Device table
        tab_devices = tk.Frame(nb, bg=C_BG)
        nb.add(tab_devices, text="  ⬡ Device Table  ")
        self._build_device_table(tab_devices)

        # Tab 2: Flagged
        tab_flagged = tk.Frame(nb, bg=C_BG)
        nb.add(tab_flagged, text="  ⚠ Flagged  ")
        self._build_flagged_table(tab_flagged)

        # Tab 3: Breach window
        tab_breach = tk.Frame(nb, bg=C_BG)
        nb.add(tab_breach, text="  ⏱ Breach Window  ")
        self._build_breach_table(tab_breach)

        # Tab 4: Log
        tab_log = tk.Frame(nb, bg=C_BG)
        nb.add(tab_log, text="  ☰ Analysis Log  ")
        self._build_log(tab_log)

    def _make_tree(self, parent, columns, headings, widths):
        frame = tk.Frame(parent, bg=C_BG)
        frame.pack(fill="both", expand=True, padx=4, pady=4)
        tree = ttk.Treeview(frame, columns=columns, show="headings",
                            selectmode="browse")
        for col, hdr, w in zip(columns, headings, widths):
            tree.heading(col, text=hdr)
            tree.column(col,  width=w, minwidth=40, anchor="w")
        vsb = ttk.Scrollbar(frame, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid (row=0, column=1, sticky="ns")
        hsb.grid (row=1, column=0, sticky="ew")
        frame.rowconfigure(0,    weight=1)
        frame.columnconfigure(0, weight=1)
        tree.tag_configure("flagged", foreground=C_DANGER, background="#1A0D0D")
        tree.tag_configure("clean",   foreground=C_TEXT)
        return tree

    def _build_device_table(self, parent):
        cols = ("idx","serial","friendly","vid","pid","first","last","drive","anomaly")
        hdrs = ("#","Serial / ID","Friendly name","VID","PID","First seen","Last seen","Drive","Status")
        wids = (30, 180, 220, 60, 60, 150, 150, 55, 140)
        self.tree_all = self._make_tree(parent, cols, hdrs, wids)

    def _build_flagged_table(self, parent):
        cols = ("serial","friendly","vid","pid","first","anomaly")
        hdrs = ("Serial / ID","Friendly name","VID","PID","First seen","Anomaly flags")
        wids = (200, 230, 70, 70, 160, 200)
        self.tree_flagged = self._make_tree(parent, cols, hdrs, wids)

    def _build_breach_table(self, parent):
        # Header strip
        hdr = tk.Frame(parent, bg=C_HEADER_BG, height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Devices connected within breach window",
                 bg=C_HEADER_BG, fg=C_WARN,
                 font=("Courier New", 9, "bold")).pack(side="left", padx=12, pady=8)
        cols = ("serial","friendly","vid","pid","first","drive","anomaly")
        hdrs = ("Serial / ID","Friendly name","VID","PID","First seen","Drive","Status")
        wids = (200, 240, 70, 70, 160, 55, 140)
        self.tree_breach = self._make_tree(parent, cols, hdrs, wids)

    def _build_log(self, parent):
        f = tk.Frame(parent, bg=C_BG)
        f.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_text = tk.Text(f, bg=C_PANEL, fg=C_TEXT,
                                font=("Courier New", 9), relief="flat",
                                insertbackground=C_ACCENT, wrap="word",
                                padx=10, pady=10)
        vsb = ttk.Scrollbar(f, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=vsb.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        f.rowconfigure(0, weight=1)
        f.columnconfigure(0, weight=1)
        self.log_text.tag_configure("info",    foreground=C_ACCENT)
        self.log_text.tag_configure("warn",    foreground=C_WARN)
        self.log_text.tag_configure("danger",  foreground=C_DANGER)
        self.log_text.tag_configure("success", foreground=C_SUCCESS)
        self.log_text.tag_configure("muted",   foreground=C_MUTED)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _log(self, msg, level="info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{ts}]  ", "muted")
        self.log_text.insert("end", msg + "\n", level)
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _set_status(self, msg, color=C_SUCCESS):
        self.status_var.set(msg)
        for w in self.nametowidget(".").winfo_children():
            pass  # just update
        self.update_idletasks()

    def _update_stats(self):
        total   = len(self.devices)
        flagged = sum(1 for d in self.devices if d.get("anomaly","clean") != "clean")
        clean   = total - flagged
        window  = len(filter_by_window(self.devices, self.v_start.get(), self.v_end.get()))
        self.lbl_total["text"]   = str(total)
        self.lbl_clean["text"]   = str(clean)
        self.lbl_flagged["text"] = str(flagged)
        self.lbl_window["text"]  = str(window)

    def _populate_trees(self):
        # All devices
        self.tree_all.delete(*self.tree_all.get_children())
        for i, d in enumerate(self.devices, 1):
            tag = "flagged" if d.get("anomaly","clean") != "clean" else "clean"
            self.tree_all.insert("", "end", tags=(tag,), values=(
                i,
                d.get("serial","")[:30],
                d.get("friendly_name","")[:38],
                d.get("vid",""),
                d.get("pid",""),
                (d.get("first_install") or "")[:19],
                (d.get("last_arrival") or "")[:19],
                d.get("drive_letter",""),
                d.get("anomaly","clean"),
            ))

        # Flagged
        self.tree_flagged.delete(*self.tree_flagged.get_children())
        for d in self.devices:
            if d.get("anomaly","clean") != "clean":
                self.tree_flagged.insert("", "end", tags=("flagged",), values=(
                    d.get("serial","")[:30],
                    d.get("friendly_name","")[:38],
                    d.get("vid",""), d.get("pid",""),
                    (d.get("first_install") or "")[:19],
                    d.get("anomaly",""),
                ))

        # Breach window
        self.tree_breach.delete(*self.tree_breach.get_children())
        filtered = filter_by_window(self.devices, self.v_start.get(), self.v_end.get())
        for d in filtered:
            tag = "flagged" if d.get("anomaly","clean") != "clean" else "clean"
            self.tree_breach.insert("", "end", tags=(tag,), values=(
                d.get("serial","")[:30],
                d.get("friendly_name","")[:38],
                d.get("vid",""), d.get("pid",""),
                (d.get("first_install") or "")[:19],
                d.get("drive_letter",""),
                d.get("anomaly","clean"),
            ))

    # ── Actions ────────────────────────────────────────────────────────────────
    def _load_demo(self):
        self.devices = [dict(d) for d in DEMO_DEVICES]
        flag_anomalies(self.devices)
        self._populate_trees()
        self._update_stats()
        self._log("Loaded demo dataset (6 synthetic devices)", "info")
        self._log("3 clean, 1 flagged (unknown_vid), 1 flagged (unknown_vid + suspicious_serial), 1 flagged (unknown_vid)", "warn")
        self._set_status("Demo data loaded")

    def _run_analysis(self):
        def worker():
            self._log("Starting analysis…", "info")
            devices = []
            sys_path = self.v_system.get()
            if sys_path:
                self._log(f"Parsing USBSTOR from: {sys_path}", "info")
                found = parse_usbstor(sys_path)
                self._log(f"  → {len(found)} record(s)", "success" if found else "warn")
                devices.extend(found)
            else:
                self._log("No SYSTEM hive specified — using demo data", "warn")
                devices = [dict(d) for d in DEMO_DEVICES]

            api_path = self.v_setupapi.get()
            if api_path:
                self._log(f"Parsing SetupAPI log: {api_path}", "info")
                api_data = parse_setupapi(api_path)
                self._log(f"  → {len(api_data)} install record(s)", "success")

            flag_anomalies(devices)
            flagged = sum(1 for d in devices if d.get("anomaly","clean") != "clean")
            self.devices = devices
            self.after(0, self._populate_trees)
            self.after(0, self._update_stats)
            self.after(0, lambda: self._log(
                f"Analysis complete: {len(devices)} devices, {flagged} flagged", "success"))
            self.after(0, lambda: self._set_status(
                f"Analysis complete: {len(devices)} devices, {flagged} flagged"))

        threading.Thread(target=worker, daemon=True).start()

    def _export_csv(self):
        if not self.devices:
            messagebox.showwarning("No data", "Run analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV","*.csv")],
            initialfile="usb_report.csv")
        if not path: return
        fields = ["source","device_class","serial","friendly_name",
                  "hardware_id","vid","pid","revision",
                  "first_install","last_arrival","drive_letter","anomaly"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(self.devices)
        self._log(f"CSV exported: {path}", "success")
        messagebox.showinfo("Exported", f"CSV saved to:\n{path}")

    def _generate_report(self):
        if not self.devices:
            messagebox.showwarning("No data", "Run analysis or load demo data first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf", filetypes=[("PDF","*.pdf")],
            initialfile=f"usb_forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        if not path: return

        self._set_status("Generating PDF report…")
        self._log("Generating PDF report…", "info")

        def worker():
            try:
                filtered = filter_by_window(
                    self.devices, self.v_start.get(), self.v_end.get())
                generate_pdf_report(
                    self.devices, path,
                    analyst_name=self.v_analyst.get(),
                    case_id=self.v_case_id.get(),
                    breach_start=self.v_start.get(),
                    breach_end=self.v_end.get(),
                    filtered_devices=filtered,
                )
                self.after(0, lambda: self._log(f"PDF saved: {path}", "success"))
                self.after(0, lambda: self._set_status("PDF report generated"))
                self.after(0, lambda: messagebox.showinfo(
                    "Report Generated",
                    f"PDF report saved to:\n{path}\n\nOpen it now?") or
                    (os.startfile(path) if sys.platform == "win32" else
                     os.system(f"xdg-open '{path}'") if sys.platform == "linux" else
                     os.system(f"open '{path}'")))
            except Exception as ex:
                self.after(0, lambda: self._log(f"PDF error: {ex}", "danger"))
                self.after(0, lambda: messagebox.showerror("Error", str(ex)))

        threading.Thread(target=worker, daemon=True).start()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = USBForensicApp()
    app.mainloop()
