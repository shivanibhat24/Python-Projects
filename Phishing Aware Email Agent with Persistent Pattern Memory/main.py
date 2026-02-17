#!/usr/bin/env python3
"""
Phishing-Aware Email Agent using Model Context Protocol
Author: Shivani Bhat
Version: 2.0.0

Novel additions:
  - Header analysis (SPF/DKIM/DMARC simulation)
  - Attachment risk scanning
  - Threat intelligence feed integration
  - Async batch analysis
  - Behavioral anomaly detection (time-of-day, geo-IP hints)
  - HTML/CSS obfuscation detection
  - Whitelist / trusted-sender management
  - Structured JSON + CSV export
  - Rich CLI dashboard
  - Pluggable scorer architecture
"""

import re
import json
import csv
import hashlib
import sqlite3
import asyncio
import logging
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
from collections import defaultdict

# ── optional rich terminal output ──────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich import print as rprint
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Enums & constants ───────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    SAFE     = "SAFE"
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

def risk_level_from_score(score: float) -> RiskLevel:
    if score < 0.20:  return RiskLevel.SAFE
    if score < 0.40:  return RiskLevel.LOW
    if score < 0.60:  return RiskLevel.MEDIUM
    if score < 0.80:  return RiskLevel.HIGH
    return RiskLevel.CRITICAL

RISK_EMOJI = {
    RiskLevel.SAFE:     "✅",
    RiskLevel.LOW:      "🟡",
    RiskLevel.MEDIUM:   "🟠",
    RiskLevel.HIGH:     "🔴",
    RiskLevel.CRITICAL: "☠️",
}


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class PhishingPattern:
    pattern_type: str
    pattern_value: str
    confidence_score: float
    first_seen: str
    last_seen: str
    occurrence_count: int
    context: str

@dataclass
class AttachmentScanResult:
    filename: str
    extension: str
    mime_type: str
    risk_score: float
    risk_reasons: List[str] = field(default_factory=list)

@dataclass
class HeaderAnalysisResult:
    spf_pass: Optional[bool]          = None
    dkim_pass: Optional[bool]         = None
    dmarc_pass: Optional[bool]        = None
    received_chain_suspicious: bool   = False
    mismatched_from: bool             = False
    risk_score: float                 = 0.0

@dataclass
class EmailAnalysisResult:
    email_id: str
    risk_score: float
    risk_level: str
    detected_patterns: List[PhishingPattern]
    recommendations: List[str]
    is_suspicious: bool
    analysis_timestamp: str
    header_analysis: Optional[HeaderAnalysisResult] = None
    attachment_results: List[AttachmentScanResult]  = field(default_factory=list)
    obfuscation_detected: bool                      = False
    behavioral_flags: List[str]                     = field(default_factory=list)
    threat_intel_hits: List[str]                    = field(default_factory=list)


# ── Model Context Protocol (persistence layer) ─────────────────────────────────

class ModelContextProtocol:
    """SQLite-backed pattern memory with analytics helpers."""

    def __init__(self, db_path: str = "phishing_patterns.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.executescript('''
            CREATE TABLE IF NOT EXISTS patterns (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type    TEXT NOT NULL,
                pattern_value   TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                first_seen      TEXT NOT NULL,
                last_seen       TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                context         TEXT,
                UNIQUE(pattern_type, pattern_value)
            );

            CREATE TABLE IF NOT EXISTS email_analyses (
                email_id           TEXT PRIMARY KEY,
                risk_score         REAL NOT NULL,
                risk_level         TEXT NOT NULL,
                detected_patterns  TEXT,
                recommendations    TEXT,
                is_suspicious      BOOLEAN NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                obfuscation_detected BOOLEAN DEFAULT 0,
                behavioral_flags   TEXT,
                threat_intel_hits  TEXT
            );

            CREATE TABLE IF NOT EXISTS trusted_senders (
                domain TEXT PRIMARY KEY,
                added_by TEXT,
                added_at TEXT
            );

            CREATE TABLE IF NOT EXISTS threat_intel (
                ioc_type  TEXT NOT NULL,
                ioc_value TEXT NOT NULL,
                severity  REAL NOT NULL,
                source    TEXT,
                added_at  TEXT,
                PRIMARY KEY(ioc_type, ioc_value)
            );
        ''')
        conn.commit()
        conn.close()
        self._seed_threat_intel()

    def _seed_threat_intel(self):
        """Seed with a small built-in IOC list."""
        iocs = [
            ("domain", "paypal-security.com",      0.95, "built-in"),
            ("domain", "amazon-security.net",       0.95, "built-in"),
            ("domain", "account-verify-login.com",  0.90, "built-in"),
            ("domain", "secure-banklogin.net",      0.92, "built-in"),
            ("ip",     "192.168.1.100",             0.80, "built-in"),
            ("url",    "bit.ly",                    0.70, "built-in"),
            ("url",    "tinyurl.com",               0.65, "built-in"),
        ]
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for ioc in iocs:
            c.execute('''
                INSERT OR IGNORE INTO threat_intel (ioc_type, ioc_value, severity, source, added_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (*ioc, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ── pattern CRUD ──────────────────────────────────────────────────────────

    def store_pattern(self, pattern: PhishingPattern):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO patterns
                (pattern_type, pattern_value, confidence_score, first_seen, last_seen, occurrence_count, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (pattern.pattern_type, pattern.pattern_value, pattern.confidence_score,
                  pattern.first_seen, pattern.last_seen, pattern.occurrence_count, pattern.context))
            conn.commit()
        except Exception as e:
            logger.error(f"store_pattern error: {e}")
        finally:
            conn.close()

    def get_patterns(self, pattern_type: Optional[str] = None) -> List[PhishingPattern]:
        conn = sqlite3.connect(self.db_path)
        if pattern_type:
            rows = conn.execute('SELECT * FROM patterns WHERE pattern_type=?', (pattern_type,)).fetchall()
        else:
            rows = conn.execute('SELECT * FROM patterns').fetchall()
        conn.close()
        return [PhishingPattern(*row[1:]) for row in rows]

    def update_pattern_occurrence(self, pattern_type: str, pattern_value: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            UPDATE patterns SET occurrence_count=occurrence_count+1, last_seen=?
            WHERE pattern_type=? AND pattern_value=?
        ''', (datetime.now().isoformat(), pattern_type, pattern_value))
        conn.commit()
        conn.close()

    def store_analysis(self, result: EmailAnalysisResult):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO email_analyses
            (email_id, risk_score, risk_level, detected_patterns, recommendations,
             is_suspicious, analysis_timestamp, obfuscation_detected, behavioral_flags, threat_intel_hits)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        ''', (
            result.email_id, result.risk_score, result.risk_level,
            json.dumps([asdict(p) for p in result.detected_patterns]),
            json.dumps(result.recommendations), result.is_suspicious,
            result.analysis_timestamp, result.obfuscation_detected,
            json.dumps(result.behavioral_flags), json.dumps(result.threat_intel_hits)
        ))
        conn.commit()
        conn.close()

    # ── trusted senders ───────────────────────────────────────────────────────

    def add_trusted_sender(self, domain: str, added_by: str = "user"):
        conn = sqlite3.connect(self.db_path)
        conn.execute('INSERT OR REPLACE INTO trusted_senders VALUES (?,?,?)',
                     (domain.lower(), added_by, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    def is_trusted_sender(self, email_address: str) -> bool:
        domain = email_address.split("@")[-1].lower() if "@" in email_address else ""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute('SELECT 1 FROM trusted_senders WHERE domain=?', (domain,)).fetchone()
        conn.close()
        return row is not None

    # ── threat intel ──────────────────────────────────────────────────────────

    def lookup_ioc(self, ioc_type: str, ioc_value: str) -> Optional[float]:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            'SELECT severity FROM threat_intel WHERE ioc_type=? AND ioc_value=?',
            (ioc_type, ioc_value.lower())
        ).fetchone()
        conn.close()
        return row[0] if row else None

    def add_ioc(self, ioc_type: str, ioc_value: str, severity: float, source: str = "user"):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO threat_intel (ioc_type, ioc_value, severity, source, added_at)
            VALUES (?,?,?,?,?)
        ''', (ioc_type, ioc_value.lower(), severity, source, datetime.now().isoformat()))
        conn.commit()
        conn.close()

    # ── analytics ─────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        patterns = self.get_patterns()
        total_emails = conn.execute('SELECT COUNT(*) FROM email_analyses').fetchone()[0]
        suspicious   = conn.execute('SELECT COUNT(*) FROM email_analyses WHERE is_suspicious=1').fetchone()[0]
        avg_risk     = conn.execute('SELECT AVG(risk_score) FROM email_analyses').fetchone()[0] or 0
        conn.close()

        type_counts: Dict[str, int] = defaultdict(int)
        for p in patterns:
            type_counts[p.pattern_type] += 1

        top_patterns = sorted(patterns, key=lambda x: x.occurrence_count, reverse=True)[:10]

        return {
            "total_patterns":          len(patterns),
            "high_confidence_patterns": sum(1 for p in patterns if p.confidence_score > 0.8),
            "pattern_types":           dict(type_counts),
            "total_emails_analyzed":   total_emails,
            "suspicious_emails":       suspicious,
            "average_risk_score":      round(avg_risk, 3),
            "most_common_patterns": [
                {"type": p.pattern_type,
                 "value": (p.pattern_value[:50] + "...") if len(p.pattern_value) > 50 else p.pattern_value,
                 "count": p.occurrence_count,
                 "confidence": round(p.confidence_score, 2)}
                for p in top_patterns
            ]
        }

    # ── export ────────────────────────────────────────────────────────────────

    def export_analyses(self, output_path: str, fmt: str = "json"):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute('SELECT * FROM email_analyses').fetchall()
        headers = [d[0] for d in conn.execute('SELECT * FROM email_analyses LIMIT 0').description or []]
        conn.close()

        if fmt == "json":
            data = [dict(zip(headers, r)) for r in rows]
            Path(output_path).write_text(json.dumps(data, indent=2))
        elif fmt == "csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                if headers:
                    writer.writerow(headers)
                writer.writerows(rows)
        logger.info(f"Exported {len(rows)} analyses to {output_path} ({fmt})")


# ── Pluggable scorer interface ─────────────────────────────────────────────────

class BaseScorer:
    """Extend this to add custom scoring plugins."""
    weight: float = 0.0

    def score(self, email_data: Dict[str, Any]) -> Tuple[float, List[PhishingPattern]]:
        return 0.0, []


# ── Detection Engine ───────────────────────────────────────────────────────────

class PhishingDetector:
    """Core detection engine with extended analysers."""

    SUSPICIOUS_DOMAINS = {
        'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl',
        'paypal-security.com', 'amazon-security.net',
        'ow.ly', 'is.gd', 'buff.ly', 'adf.ly',
    }

    URGENT_KEYWORDS = [
        'urgent', 'immediate action', 'verify now', 'click here',
        'suspended', 'limited time', 'expires today', 'act now',
        'confirm your identity', 'account will be closed',
        'unauthorized access', 'security alert', 'one-time password',
    ]

    RISKY_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.scr', '.pif', '.vbs', '.js',
        '.jar', '.ps1', '.hta', '.msi', '.dll', '.reg',
    }

    DOUBLE_EXTENSION_PATTERN = re.compile(
        r'\.(pdf|docx?|xlsx?|zip)\.(exe|bat|scr|vbs|js)$', re.IGNORECASE
    )

    HTML_OBFUSCATION_PATTERNS = [
        re.compile(r'&#[xX]?[0-9a-fA-F]+;'),           # HTML entities
        re.compile(r'\\u[0-9a-fA-F]{4}'),               # Unicode escapes
        re.compile(r'eval\s*\(', re.IGNORECASE),         # eval() JS
        re.compile(r'document\.write\s*\(', re.IGNORECASE),
        re.compile(r'unescape\s*\(', re.IGNORECASE),
        re.compile(r'String\.fromCharCode', re.IGNORECASE),
    ]

    # ── URL analysis ──────────────────────────────────────────────────────────

    def analyze_urls(self, text: str) -> List[Tuple[str, float, str]]:
        """Returns (url, risk_score, reason)."""
        results = []
        url_re = re.compile(
            r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', re.IGNORECASE
        )
        ip_re  = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')
        brand_re = re.compile(
            r'[a-zA-Z0-9-]+\.(paypal|amazon|microsoft|google|apple|netflix|irs|fedex|dhl)\.',
            re.IGNORECASE
        )
        homoglyph_re = re.compile(r'[аеіоурсхАЕІОУРСХ]')   # Cyrillic lookalikes

        for url in url_re.findall(text):
            score = 0.0
            reasons = []

            if any(d in url.lower() for d in self.SUSPICIOUS_DOMAINS):
                score += 0.70; reasons.append("known suspicious domain/shortener")

            if ip_re.search(url):
                score += 0.90; reasons.append("raw IP address used instead of domain")

            if brand_re.search(url):
                score += 0.85; reasons.append("brand name used in suspicious subdomain")

            if homoglyph_re.search(url):
                score += 0.80; reasons.append("homoglyph/IDN characters detected")

            # Excessive subdomains
            try:
                from urllib.parse import urlparse
                hostname = urlparse(url).hostname or ""
                if hostname.count('.') >= 4:
                    score += 0.50; reasons.append("excessive subdomain nesting")
            except Exception:
                pass

            # Port in URL (non-standard)
            if re.search(r':\d{4,5}/', url):
                score += 0.40; reasons.append("non-standard port in URL")

            if score > 0.25:
                results.append((url, min(score, 1.0), "; ".join(reasons)))

        return results

    # ── content analysis ──────────────────────────────────────────────────────

    def analyze_content(self, text: str) -> List[Tuple[str, str, float]]:
        indicators = []
        tl = text.lower()

        for kw in self.URGENT_KEYWORDS:
            if kw in tl:
                indicators.append(('urgent_language', kw, 0.55))

        for pat in [
            r'verify.{0,25}(password|account|identity)',
            r'confirm.{0,25}(login|credentials|details)',
            r'update.{0,25}(payment|billing|card|ssn)',
        ]:
            for m in re.findall(pat, text, re.IGNORECASE):
                indicators.append(('credential_harvesting', str(m), 0.80))

        for pat in [
            r'congratulations.{0,60}(won|winner|selected|chosen)',
            r'you.{0,15}have.{0,15}been.{0,15}chosen',
            r'claim.{0,25}(prize|reward|bonus|gift)',
        ]:
            for m in re.findall(pat, text, re.IGNORECASE):
                indicators.append(('social_engineering', str(m), 0.70))

        # Excessive caps (≥ 40 % uppercase = shouting / alarm)
        letters = [c for c in text if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.40:
            indicators.append(('excessive_caps', 'high uppercase ratio', 0.40))

        return indicators

    # ── HTML/CSS obfuscation ──────────────────────────────────────────────────

    def detect_obfuscation(self, html_body: str) -> Tuple[bool, List[str]]:
        hits = []
        for pat in self.HTML_OBFUSCATION_PATTERNS:
            if pat.search(html_body):
                hits.append(pat.pattern)
        return bool(hits), hits

    # ── sender analysis ───────────────────────────────────────────────────────

    def analyze_sender(self, sender: str, reply_to: str = "") -> List[Tuple[str, str, float]]:
        indicators = []
        spoofing_indicators = [
            'noreply', 'no-reply', 'automated', 'system',
            'security-team', 'account-verification', 'do-not-reply',
        ]
        for ind in spoofing_indicators:
            if ind in sender.lower():
                indicators.append(('sender_spoofing', ind, 0.55))

        if reply_to and sender != reply_to:
            indicators.append(('sender_reply_to_mismatch', f"{sender} vs {reply_to}", 0.70))

        # Display-name / envelope mismatch  (e.g.  "PayPal" <bad@evil.com>)
        display_name_re = re.compile(r'"?([^"<]+)"?\s+<([^>]+)>', re.IGNORECASE)
        m = display_name_re.match(sender)
        if m:
            display, envelope = m.group(1).strip(), m.group(2).strip()
            legit_brands = ['paypal', 'amazon', 'microsoft', 'google', 'apple', 'netflix']
            for brand in legit_brands:
                if brand in display.lower() and brand not in envelope.lower():
                    indicators.append(('display_name_spoofing', f'"{display}" from {envelope}', 0.90))
                    break

        return indicators

    # ── email header analysis ─────────────────────────────────────────────────

    def analyze_headers(self, headers: Dict[str, str]) -> HeaderAnalysisResult:
        result = HeaderAnalysisResult()

        auth_results = headers.get("Authentication-Results", "").lower()
        if auth_results:
            result.spf_pass  = "spf=pass"  in auth_results
            result.dkim_pass = "dkim=pass" in auth_results
            result.dmarc_pass= "dmarc=pass"in auth_results
            if not result.spf_pass:  result.risk_score += 0.30
            if not result.dkim_pass: result.risk_score += 0.25
            if not result.dmarc_pass:result.risk_score += 0.20

        from_header    = headers.get("From", "")
        return_path    = headers.get("Return-Path", "")
        if from_header and return_path:
            from_domain = from_header.split("@")[-1].strip(">").lower()
            rp_domain   = return_path.split("@")[-1].strip(">").lower()
            if from_domain and rp_domain and from_domain != rp_domain:
                result.mismatched_from = True
                result.risk_score += 0.35

        received = headers.get("Received", "")
        if received.count("Received:") > 6:
            result.received_chain_suspicious = True
            result.risk_score += 0.20

        result.risk_score = min(result.risk_score, 1.0)
        return result

    # ── attachment scanning ───────────────────────────────────────────────────

    def scan_attachment(self, filename: str, mime_type: str = "") -> AttachmentScanResult:
        ext  = Path(filename).suffix.lower()
        risk = 0.0
        reasons: List[str] = []

        if ext in self.RISKY_EXTENSIONS:
            risk += 0.90; reasons.append(f"dangerous extension: {ext}")

        if self.DOUBLE_EXTENSION_PATTERN.search(filename):
            risk += 0.95; reasons.append("double extension camouflage")

        if mime_type and ext:
            expected = {
                ".pdf":  "application/pdf",
                ".docx": "application/vnd.openxmlformats",
                ".xlsx": "application/vnd.openxmlformats",
                ".zip":  "application/zip",
            }
            exp = expected.get(ext)
            if exp and exp not in mime_type:
                risk += 0.60; reasons.append("MIME type mismatch")

        # Password-protected archives — common malware delivery
        if ext == ".zip" and "password" in filename.lower():
            risk += 0.50; reasons.append("password-protected archive")

        return AttachmentScanResult(
            filename=filename, extension=ext, mime_type=mime_type,
            risk_score=min(risk, 1.0), risk_reasons=reasons
        )

    # ── behavioral anomaly ────────────────────────────────────────────────────

    def detect_behavioral_anomalies(self, email_data: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        ts_str = email_data.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.hour < 5 or ts.hour > 23:
                flags.append(f"Email sent at unusual hour ({ts.hour:02d}:{ts.minute:02d})")
        except Exception:
            pass

        # Multi-language body (simple heuristic)
        body = email_data.get("body", "")
        cjk  = re.search(r'[\u4e00-\u9fff\u3040-\u30ff]', body)
        if cjk:
            flags.append("CJK characters in body — possible multi-language obfuscation")

        return flags


# ── Main Agent ─────────────────────────────────────────────────────────────────

class PhishingAwareEmailAgent:
    """
    Orchestrates detection, learning, whitelist management,
    and threat intelligence for phishing-aware email analysis.
    """

    def __init__(self, db_path: str = "phishing_patterns.db",
                 risk_threshold: float = 0.45,
                 extra_scorers: Optional[List[BaseScorer]] = None):
        self.mcp              = ModelContextProtocol(db_path)
        self.detector         = PhishingDetector()
        self.risk_threshold   = risk_threshold
        self.extra_scorers    = extra_scorers or []

    # ── single email analysis ─────────────────────────────────────────────────

    def analyze_email(self, email_data: Dict[str, Any]) -> EmailAnalysisResult:
        email_id  = self._generate_email_id(email_data)
        patterns: List[PhishingPattern] = []
        risk_score = 0.0
        threat_intel_hits: List[str] = []

        sender   = email_data.get("sender",   "")
        subject  = email_data.get("subject",  "")
        body     = email_data.get("body",     "")
        reply_to = email_data.get("reply_to", "")
        headers  = email_data.get("headers",  {})
        attachments = email_data.get("attachments", [])  # list of {"filename":..,"mime":..}

        now = datetime.now().isoformat()

        # ── 0. Whitelist shortcut ─────────────────────────────────────────────
        if self.mcp.is_trusted_sender(sender):
            return EmailAnalysisResult(
                email_id=email_id, risk_score=0.0, risk_level=RiskLevel.SAFE.value,
                detected_patterns=[], recommendations=["Sender is whitelisted — trusted."],
                is_suspicious=False, analysis_timestamp=now,
            )

        # ── 1. Threat intelligence check ─────────────────────────────────────
        domains_in_body = re.findall(r'https?://([^\s/<>"\']+)', body, re.IGNORECASE)
        for domain in set(domains_in_body):
            sev = self.mcp.lookup_ioc("domain", domain.lower())
            if sev:
                risk_score += sev * 0.40
                threat_intel_hits.append(f"IOC domain match: {domain} (severity {sev:.2f})")

        sender_domain = sender.split("@")[-1].lower() if "@" in sender else sender.lower()
        sev = self.mcp.lookup_ioc("domain", sender_domain)
        if sev:
            risk_score += sev * 0.45
            threat_intel_hits.append(f"IOC sender domain: {sender_domain} (severity {sev:.2f})")

        # ── 2. URL analysis ───────────────────────────────────────────────────
        for url, score, reason in self.detector.analyze_urls(body):
            p = PhishingPattern("suspicious_url", url, score, now, now, 1, reason)
            patterns.append(p); risk_score += score * 0.28

        # ── 3. Content analysis ───────────────────────────────────────────────
        for itype, val, score in self.detector.analyze_content(subject + " " + body):
            p = PhishingPattern(itype, val, score, now, now, 1, "content scan")
            patterns.append(p); risk_score += score * 0.22

        # ── 4. Sender analysis ────────────────────────────────────────────────
        for itype, val, score in self.detector.analyze_sender(sender, reply_to):
            p = PhishingPattern(itype, val, score, now, now, 1, "sender scan")
            patterns.append(p); risk_score += score * 0.20

        # ── 5. Header analysis ────────────────────────────────────────────────
        header_result = None
        if headers:
            header_result = self.detector.analyze_headers(headers)
            risk_score += header_result.risk_score * 0.25

        # ── 6. Attachment scanning ────────────────────────────────────────────
        attach_results: List[AttachmentScanResult] = []
        for att in attachments:
            ar = self.detector.scan_attachment(att.get("filename",""), att.get("mime",""))
            attach_results.append(ar)
            risk_score += ar.risk_score * 0.35

        # ── 7. Obfuscation detection ──────────────────────────────────────────
        obfuscated, _ = self.detector.detect_obfuscation(body)

        # ── 8. Behavioral anomalies ───────────────────────────────────────────
        behavioral_flags = self.detector.detect_behavioral_anomalies(email_data)
        risk_score += len(behavioral_flags) * 0.05

        # ── 9. Known-pattern boost ────────────────────────────────────────────
        self._check_known_patterns(patterns)

        # ── 10. Extra scorer plugins ──────────────────────────────────────────
        for scorer in self.extra_scorers:
            extra_score, extra_patterns = scorer.score(email_data)
            risk_score += extra_score * scorer.weight
            patterns.extend(extra_patterns)

        # ── Finalize ──────────────────────────────────────────────────────────
        risk_score = min(risk_score, 1.0)
        level      = risk_level_from_score(risk_score)

        recommendations = self._generate_recommendations(patterns, risk_score,
                                                          obfuscated, attach_results,
                                                          threat_intel_hits)
        for p in patterns:
            self._store_or_update_pattern(p)

        result = EmailAnalysisResult(
            email_id=email_id,
            risk_score=round(risk_score, 4),
            risk_level=level.value,
            detected_patterns=patterns,
            recommendations=recommendations,
            is_suspicious=risk_score > self.risk_threshold,
            analysis_timestamp=now,
            header_analysis=header_result,
            attachment_results=attach_results,
            obfuscation_detected=obfuscated,
            behavioral_flags=behavioral_flags,
            threat_intel_hits=threat_intel_hits,
        )
        self.mcp.store_analysis(result)
        return result

    # ── batch analysis ────────────────────────────────────────────────────────

    async def analyze_batch_async(self, emails: List[Dict[str, Any]]) -> List[EmailAnalysisResult]:
        """Analyze a list of emails concurrently using asyncio."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self.analyze_email, e) for e in emails]
        return await asyncio.gather(*tasks)

    def analyze_batch(self, emails: List[Dict[str, Any]]) -> List[EmailAnalysisResult]:
        return asyncio.run(self.analyze_batch_async(emails))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _generate_email_id(self, email_data: Dict[str, Any]) -> str:
        content = f"{email_data.get('sender','')}{email_data.get('subject','')}{email_data.get('timestamp','')}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _check_known_patterns(self, current: List[PhishingPattern]):
        known = self.mcp.get_patterns()
        for cp in current:
            for kp in known:
                if cp.pattern_type == kp.pattern_type and \
                   cp.pattern_value.lower() in kp.pattern_value.lower():
                    cp.confidence_score = min(cp.confidence_score + 0.10, 1.0)
                    self.mcp.update_pattern_occurrence(kp.pattern_type, kp.pattern_value)

    def _store_or_update_pattern(self, pattern: PhishingPattern):
        existing = self.mcp.get_patterns(pattern.pattern_type)
        for ex in existing:
            if ex.pattern_value == pattern.pattern_value:
                ex.occurrence_count += 1
                ex.last_seen = datetime.now().isoformat()
                ex.confidence_score = min(ex.confidence_score + 0.05, 1.0)
                self.mcp.store_pattern(ex)
                return
        self.mcp.store_pattern(pattern)

    def _generate_recommendations(
        self,
        patterns: List[PhishingPattern],
        risk_score: float,
        obfuscated: bool,
        attachments: List[AttachmentScanResult],
        intel_hits: List[str],
    ) -> List[str]:
        recs: List[str] = []
        level = risk_level_from_score(risk_score)

        level_msgs = {
            RiskLevel.SAFE:     "Email appears safe. Stay vigilant as a general practice.",
            RiskLevel.LOW:      "LOW RISK — verify any links before clicking.",
            RiskLevel.MEDIUM:   "MEDIUM RISK — exercise caution; verify sender independently.",
            RiskLevel.HIGH:     "HIGH RISK — do not interact; report to IT/security.",
            RiskLevel.CRITICAL: "CRITICAL — delete immediately and report. Do NOT click anything.",
        }
        recs.append(f"{RISK_EMOJI[level]} {level_msgs[level]}")

        types = {p.pattern_type for p in patterns}
        if 'suspicious_url' in types:
            recs.append("Hover over all links to verify destinations before clicking.")
        if 'credential_harvesting' in types:
            recs.append("Never enter passwords or card details via email links.")
        if 'sender_spoofing' in types or 'display_name_spoofing' in types:
            recs.append("Sender identity appears spoofed — contact the organisation via official website.")
        if 'urgent_language' in types:
            recs.append("Artificial urgency is a hallmark of phishing — take time to verify.")
        if obfuscated:
            recs.append("HTML obfuscation detected — this email hides its true content.")
        for ar in attachments:
            if ar.risk_score > 0.5:
                recs.append(f"Do NOT open attachment '{ar.filename}': {', '.join(ar.risk_reasons)}")
        if intel_hits:
            recs.append(f"Threat intelligence match: {intel_hits[0]}")

        return recs

    # ── public API helpers ────────────────────────────────────────────────────

    def add_trusted_sender(self, domain: str):
        self.mcp.add_trusted_sender(domain)
        logger.info(f"Trusted sender added: {domain}")

    def add_ioc(self, ioc_type: str, value: str, severity: float = 0.90):
        self.mcp.add_ioc(ioc_type, value, severity)
        logger.info(f"IOC added: [{ioc_type}] {value} (severity {severity})")

    def export(self, path: str, fmt: str = "json"):
        self.mcp.export_analyses(path, fmt)


# ── Rich CLI dashboard ─────────────────────────────────────────────────────────

def print_result(result: EmailAnalysisResult):
    level = result.risk_level
    emoji = RISK_EMOJI.get(RiskLevel(level), "❓")

    if RICH:
        color_map = {
            "SAFE": "green", "LOW": "yellow",
            "MEDIUM": "dark_orange", "HIGH": "red", "CRITICAL": "bold red"
        }
        color = color_map.get(level, "white")

        table = Table(title=f"{emoji} Email Analysis — {result.email_id}", show_lines=True)
        table.add_column("Field", style="bold cyan", width=22)
        table.add_column("Value")
        table.add_row("Risk Score",   f"[{color}]{result.risk_score:.4f}[/{color}]")
        table.add_row("Risk Level",   f"[{color}]{level}[/{color}]")
        table.add_row("Suspicious",   "YES ⚠️" if result.is_suspicious else "NO ✅")
        table.add_row("Obfuscation",  "Detected 🚨" if result.obfuscation_detected else "None")
        table.add_row("Patterns",     str(len(result.detected_patterns)))
        table.add_row("Intel Hits",   str(len(result.threat_intel_hits)))
        table.add_row("Behavior Flags", str(len(result.behavioral_flags)))
        console.print(table)

        if result.recommendations:
            console.print(Panel("\n".join(result.recommendations), title="Recommendations", border_style=color))
    else:
        print(f"\n{'─'*60}")
        print(f"{emoji} Email ID : {result.email_id}")
        print(f"   Risk     : {result.risk_score:.4f} ({level})")
        print(f"   Suspicious: {'YES' if result.is_suspicious else 'NO'}")
        print(f"   Patterns : {len(result.detected_patterns)}")
        for rec in result.recommendations:
            print(f"   ▶ {rec}")


def print_stats(stats: Dict[str, Any]):
    if RICH:
        table = Table(title="📊 Pattern Learning Statistics", show_lines=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value",  style="green")
        table.add_row("Total patterns",         str(stats["total_patterns"]))
        table.add_row("High-confidence (>0.8)", str(stats["high_confidence_patterns"]))
        table.add_row("Emails analysed",        str(stats["total_emails_analyzed"]))
        table.add_row("Suspicious emails",      str(stats["suspicious_emails"]))
        table.add_row("Average risk score",     str(stats["average_risk_score"]))
        console.print(table)

        if stats["most_common_patterns"]:
            pt = Table(title="Top Patterns")
            for col in ("Type", "Value", "Count", "Confidence"):
                pt.add_column(col)
            for p in stats["most_common_patterns"]:
                pt.add_row(p["type"], p["value"], str(p["count"]), f"{p['confidence']:.2f}")
            console.print(pt)
    else:
        print("\n=== Statistics ===")
        for k, v in stats.items():
            if k != "most_common_patterns":
                print(f"  {k}: {v}")


# ── Demo / entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phishing-Aware Email Agent v2")
    parser.add_argument("--db",     default="phishing_patterns.db", help="SQLite database path")
    parser.add_argument("--export", default="",                      help="Export results to file")
    parser.add_argument("--fmt",    default="json",                  choices=["json","csv"])
    parser.add_argument("--threshold", type=float, default=0.45,    help="Risk threshold (0–1)")
    args = parser.parse_args()

    agent = PhishingAwareEmailAgent(db_path=args.db, risk_threshold=args.threshold)

    # Add a trusted sender example
    agent.add_trusted_sender("legitimate-company.com")

    test_emails = [
        # ── Critical phishing ─────────────────────────────────────────────────
        {
            "sender":   '"PayPal Security" <security-team@paypal-security.com>',
            "subject":  "URGENT: Verify your account immediately",
            "body":     "Your PayPal account has been SUSPENDED. CLICK HERE NOW to verify: http://bit.ly/paypal-verify or your account will be deleted.",
            "reply_to": "noreply@suspicious-domain.com",
            "timestamp": "2025-03-15T02:17:00",
            "headers": {
                "Authentication-Results": "spf=fail; dkim=fail; dmarc=fail",
                "Return-Path": "<bounce@evil.com>",
                "From": "PayPal Security <security-team@paypal-security.com>",
            },
            "attachments": [{"filename": "Invoice.pdf.exe", "mime": "application/octet-stream"}],
        },
        # ── Trusted (whitelisted) ─────────────────────────────────────────────
        {
            "sender":   "newsletter@legitimate-company.com",
            "subject":  "Monthly Newsletter — March 2025",
            "body":     "Thank you for subscribing. Here are this month's product updates.",
            "reply_to": "",
            "timestamp": datetime.now().isoformat(),
            "headers": {},
            "attachments": [],
        },
        # ── Lottery scam ──────────────────────────────────────────────────────
        {
            "sender":   "winner@lottery-prize.net",
            "subject":  "Congratulations! You've won $1,000,000",
            "body":     "You have been selected as our lucky winner! Claim your prize now: http://192.168.1.100/claim",
            "reply_to": "",
            "timestamp": datetime.now().isoformat(),
            "headers": {"Authentication-Results": "spf=pass; dkim=fail"},
            "attachments": [{"filename": "claim_form.zip", "mime": "application/zip"}],
        },
        # ── Subtle spear-phishing ─────────────────────────────────────────────
        {
            "sender":   '"IT Support" <it-support@c0mpany.com>',
            "subject":  "Action required: Reset your Microsoft password",
            "body":     'Please confirm your login at https://microsoft.c0mpany-login.com to prevent account closure.',
            "reply_to": "harvester@attackerinfra.com",
            "timestamp": datetime.now().isoformat(),
            "headers": {
                "Authentication-Results": "spf=pass; dkim=pass; dmarc=fail",
                "Return-Path": "<bounce@c0mpany.com>",
                "From": "IT Support <it-support@c0mpany.com>",
            },
            "attachments": [],
        },
    ]

    if RICH:
        console.rule("[bold blue]🛡️  Phishing-Aware Email Agent v2.0  🛡️")
    else:
        print("=== Phishing-Aware Email Agent v2.0 ===\n")

    results = agent.analyze_batch(test_emails)
    for result in results:
        print_result(result)

    print_stats(agent.mcp.get_statistics())

    if args.export:
        agent.export(args.export, args.fmt)
        print(f"\nResults exported to: {args.export}")


if __name__ == "__main__":
    main()
