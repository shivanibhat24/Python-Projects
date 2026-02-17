# 🛡️ Phishing-Aware Email Agent v2.0

**Author:** Shivani Bhat  
**Version:** 2.0.0  
**Language:** Python 3.9+

An intelligent, self-learning email security agent that detects phishing attempts using multi-layer analysis, threat intelligence, and a persistent pattern memory powered by a Model Context Protocol (MCP) abstraction over SQLite.

---

## ✨ What's New in v2.0

| Feature | v1 | v2 |
|---|---|---|
| URL analysis | ✅ Basic | ✅ + homoglyphs, IDN, port anomalies |
| Content analysis | ✅ Basic | ✅ + caps ratio, extended patterns |
| Sender analysis | ✅ Basic | ✅ + display-name spoofing detection |
| **Header analysis (SPF/DKIM/DMARC)** | ❌ | ✅ |
| **Attachment scanning** | ❌ | ✅ (extension, MIME, double-ext) |
| **Threat intelligence feed** | ❌ | ✅ (seeded IOC DB + user-extensible) |
| **HTML/CSS obfuscation detection** | ❌ | ✅ |
| **Behavioral anomaly detection** | ❌ | ✅ (time-of-day, CJK obfuscation) |
| **Whitelist / trusted-sender management** | ❌ | ✅ |
| **Async batch analysis** | ❌ | ✅ (asyncio) |
| **Pluggable scorer architecture** | ❌ | ✅ (BaseScorer) |
| **Structured export (JSON + CSV)** | ❌ | ✅ |
| **Risk level enum** (SAFE→CRITICAL) | ❌ | ✅ |
| **Rich CLI dashboard** | ❌ | ✅ (optional `rich` library) |
| Email ID hashing | MD5 | SHA-256 |

---

## 🏗️ Architecture

```
PhishingAwareEmailAgent
│
├── ModelContextProtocol (SQLite persistence)
│   ├── patterns          — learned phishing patterns
│   ├── email_analyses    — full analysis history
│   ├── trusted_senders   — whitelist
│   └── threat_intel      — IOC database (domains, IPs, URLs)
│
├── PhishingDetector (stateless engine)
│   ├── analyze_urls()         — shorteners, raw IPs, homoglyphs, brand spoofing
│   ├── analyze_content()      — urgency, credential harvesting, social engineering
│   ├── analyze_sender()       — spoofing, reply-to mismatch, display-name fraud
│   ├── analyze_headers()      — SPF / DKIM / DMARC / return-path
│   ├── scan_attachment()      — risky extensions, double-extension, MIME mismatch
│   ├── detect_obfuscation()   — HTML entity abuse, eval(), fromCharCode
│   └── detect_behavioral_anomalies() — odd send time, CJK characters
│
└── BaseScorer (plugin interface)
    └── Extend to add custom ML or rule-based scoring modules
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install rich          # optional — enables the pretty CLI dashboard
```

> The agent works without `rich`; it just uses plain print output.

### 2. Run the demo

```bash
python phishing_agent.py
```

The demo automatically analyses four test emails (critical phishing, trusted sender, lottery scam, spear-phishing) and prints a colour-coded risk dashboard.

### 3. CLI options

```bash
python phishing_agent.py \
  --db      my_patterns.db   # custom database path (default: phishing_patterns.db)
  --threshold 0.45           # risk score threshold (default: 0.45)
  --export  results.json     # export analysis history
  --fmt     csv              # export format: json (default) or csv
```

---

## 📖 API Reference

### `PhishingAwareEmailAgent`

```python
agent = PhishingAwareEmailAgent(
    db_path="phishing_patterns.db",
    risk_threshold=0.45,       # flag as suspicious above this score
    extra_scorers=[]           # list of BaseScorer plugins
)
```

#### Analyse a single email

```python
result = agent.analyze_email({
    "sender":      "security@paypal-security.com",
    "subject":     "URGENT: Verify your account",
    "body":        "Click here: http://bit.ly/verify",
    "reply_to":    "attacker@evil.com",
    "timestamp":   "2025-03-15T02:17:00",
    "headers": {
        "Authentication-Results": "spf=fail; dkim=fail; dmarc=fail",
        "Return-Path": "<bounce@evil.com>",
        "From": "PayPal <security@paypal-security.com>",
    },
    "attachments": [
        {"filename": "Invoice.pdf.exe", "mime": "application/octet-stream"}
    ],
})

print(result.risk_score)    # 0.0 – 1.0
print(result.risk_level)    # SAFE | LOW | MEDIUM | HIGH | CRITICAL
print(result.is_suspicious) # True / False
```

#### Analyse a batch asynchronously

```python
results = agent.analyze_batch(list_of_email_dicts)
```

#### Manage the whitelist

```python
agent.add_trusted_sender("mycompany.com")
```

#### Add custom threat intelligence

```python
agent.add_ioc("domain", "evil-phish.net", severity=0.95)
agent.add_ioc("ip",     "10.0.0.99",      severity=0.80)
```

#### Export results

```python
agent.export("report.json", fmt="json")
agent.export("report.csv",  fmt="csv")
```

---

## 🔌 Writing a Custom Scorer Plugin

```python
from phishing_agent import BaseScorer, PhishingPattern
from datetime import datetime

class MyMLScorer(BaseScorer):
    weight = 0.30   # how much this scorer contributes to final risk_score

    def score(self, email_data):
        # integrate your ML model or rule engine here
        body = email_data.get("body", "")
        if "bitcoin wallet" in body.lower():
            p = PhishingPattern(
                pattern_type="crypto_scam",
                pattern_value="bitcoin wallet",
                confidence_score=0.85,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                occurrence_count=1,
                context="custom scorer"
            )
            return 0.85, [p]
        return 0.0, []

agent = PhishingAwareEmailAgent(extra_scorers=[MyMLScorer()])
```

---

## 🗂️ Database Schema

```sql
patterns          — accumulated phishing indicators with confidence scores
email_analyses    — full history of every email analysed
trusted_senders   — whitelisted sender domains
threat_intel      — indicator-of-compromise (IOC) feed (domain, ip, url)
```

---

## 🔬 Detection Layers Explained

### 1 · Threat Intelligence
Checks sender domain and all URLs in the body against the built-in IOC database before any other analysis. Matching IOCs heavily boost the risk score.

### 2 · URL Analysis
Flags URL shorteners, raw IP addresses, excessive subdomain nesting, non-standard ports, brand-name subdomains (`paypal.evil.com`), and Cyrillic/homoglyph lookalike characters.

### 3 · Content Analysis
Scores urgency keywords, credential-harvesting phrases, lottery/prize social-engineering language, and excessive uppercase ratios.

### 4 · Sender Analysis
Detects no-reply spoofing patterns, sender ↔ reply-to mismatches, and display-name fraud (e.g. `"PayPal" <attacker@evil.com>`).

### 5 · Header Analysis
Parses `Authentication-Results` for SPF, DKIM, and DMARC failures. Flags `From` ↔ `Return-Path` domain mismatches and suspiciously long `Received` chains.

### 6 · Attachment Scanning
Flags dangerous extensions (`.exe`, `.bat`, `.vbs`, etc.), double-extension camouflage (`Invoice.pdf.exe`), MIME-type mismatches, and password-protected archives.

### 7 · Obfuscation Detection
Searches the HTML body for encoded characters (`&#x68;`), Unicode escapes, `eval()`, `document.write()`, `unescape()`, and `String.fromCharCode` — common techniques used to hide malicious payloads.

### 8 · Behavioral Anomaly Detection
Flags emails sent between midnight and 05:00 (unusual business hours), and CJK character injection as a possible obfuscation signal.

### 9 · Pattern Memory & Learning
Every detected pattern is stored in SQLite. On subsequent emails, known patterns boost confidence scores, and occurrence counts rise — giving the agent a self-learning feedback loop.

---

## 📊 Risk Levels

| Score Range | Level    | Action |
|-------------|----------|--------|
| 0.00 – 0.19 | ✅ SAFE     | No action needed |
| 0.20 – 0.39 | 🟡 LOW      | Verify links before clicking |
| 0.40 – 0.59 | 🟠 MEDIUM   | Verify sender independently |
| 0.60 – 0.79 | 🔴 HIGH     | Do not interact; report to IT |
| 0.80 – 1.00 | ☠️ CRITICAL | Delete immediately; escalate |

---

## 🛣️ Roadmap / Ideas for Further Extension

- **Live DMARC/SPF DNS lookup** — replace the header-string simulation with real DNS resolution
- **ML classifier** — train a model on the accumulated `email_analyses` table
- **REST API** — wrap the agent with FastAPI for integration with mail servers
- **IMAP connector** — fetch and analyse live mailbox in real time
- **STIX/TAXII threat feed** — ingest enterprise-grade IOC feeds automatically
- **Attachment sandbox** — invoke ClamAV or VirusTotal API for file hash lookups

---

## 📄 License

MIT — free to use, modify, and distribute.
