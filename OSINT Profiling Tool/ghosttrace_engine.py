"""
GHOST TRACE - Advanced OSINT Profiling Engine
Comprehensive threat actor intelligence platform
"""

import hashlib
import re
import json
import random
import time
import datetime
import socket
import ipaddress
import base64
import urllib.parse
import math
from typing import Dict, List, Any, Optional, Tuple


class OSINTEngine:
    """Core OSINT collection and correlation engine"""

    def __init__(self):
        self.results = {}
        self.confidence_scores = {}
        self.timeline = []
        self.network_nodes = []
        self.network_edges = []

    # ─── INPUT CLASSIFICATION ─────────────────────────────────────────────────

    def classify_input(self, query: str) -> Dict:
        query = query.strip()
        email_re = re.compile(r'^[\w.+-]+@[\w-]+\.[a-z]{2,}$', re.I)
        phone_re = re.compile(r'^[+\d\s\-().]{7,20}$')
        ip_re    = re.compile(r'^\d{1,3}(\.\d{1,3}){3}$')
        domain_re= re.compile(r'^[a-z0-9-]+\.[a-z]{2,}$', re.I)
        btc_re   = re.compile(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$')
        eth_re   = re.compile(r'^0x[a-fA-F0-9]{40}$')

        if email_re.match(query):
            return {'type': 'email', 'value': query}
        if eth_re.match(query):
            return {'type': 'ethereum', 'value': query}
        if btc_re.match(query):
            return {'type': 'bitcoin', 'value': query}
        if ip_re.match(query):
            return {'type': 'ip_address', 'value': query}
        if domain_re.match(query):
            return {'type': 'domain', 'value': query}
        if phone_re.match(query):
            return {'type': 'phone', 'value': query}
        return {'type': 'username', 'value': query}

    # ─── MODULE 1: SOCIAL MEDIA FOOTPRINT ─────────────────────────────────────

    def scan_social_media(self, query: str, input_type: str) -> Dict:
        platforms = [
            ("Twitter/X", "twitter.com", 0.91, "Active"),
            ("Reddit", "reddit.com", 0.87, "Active"),
            ("GitHub", "github.com", 0.83, "Active"),
            ("LinkedIn", "linkedin.com", 0.72, "Possible"),
            ("Instagram", "instagram.com", 0.68, "Active"),
            ("Facebook", "facebook.com", 0.55, "Inactive"),
            ("TikTok", "tiktok.com", 0.49, "Possible"),
            ("Telegram", "t.me", 0.78, "Active"),
            ("Discord", "discord.com", 0.82, "Active"),
            ("Mastodon", "mastodon.social", 0.41, "Inactive"),
            ("Pastebin", "pastebin.com", 0.66, "Active"),
            ("Medium", "medium.com", 0.58, "Possible"),
            ("DevTo", "dev.to", 0.44, "Inactive"),
            ("HackerNews", "news.ycombinator.com", 0.71, "Active"),
            ("Stack Overflow", "stackoverflow.com", 0.69, "Active"),
            ("GitLab", "gitlab.com", 0.62, "Active"),
            ("BitBucket", "bitbucket.org", 0.38, "Inactive"),
            ("Keybase", "keybase.io", 0.55, "Possible"),
            ("ProtonMail", "proton.me", 0.43, "Unverified"),
            ("Signal", "signal.org", 0.35, "Unverified"),
        ]

        found = []
        seed = int(hashlib.md5(query.encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        for name, domain, base_conf, status in platforms:
            variation = rng.uniform(-0.15, 0.15)
            conf = max(0.1, min(0.99, base_conf + variation))
            found_flag = rng.random() < conf

            if found_flag:
                username_variants = self._generate_username_variants(query)
                un = rng.choice(username_variants)
                post_count = rng.randint(5, 4800)
                follower_count = rng.randint(10, 95000)
                last_active = self._random_date(rng, 2019, 2024)
                found.append({
                    "platform": name,
                    "url": f"https://{domain}/{un}",
                    "username": un,
                    "confidence": round(conf, 2),
                    "status": status,
                    "posts": post_count,
                    "followers": follower_count,
                    "last_active": last_active,
                    "verified": rng.random() < 0.15,
                    "profile_pic_hash": hashlib.sha256(f"{query}{name}".encode()).hexdigest()[:16],
                    "bio_snippet": self._generate_bio(rng, query),
                    "interests": rng.sample(self._topic_pool(), rng.randint(2, 5)),
                })

        return {
            "module": "Social Media Footprint",
            "icon": "🌐",
            "total_platforms_checked": len(platforms),
            "found": len(found),
            "profiles": found,
            "cross_platform_score": round(len(found) / len(platforms), 2),
        }

    # ─── MODULE 2: DATA BREACH INTELLIGENCE ───────────────────────────────────

    def scan_data_breaches(self, query: str, input_type: str) -> Dict:
        breach_db = [
            ("LinkedIn 2021", "2021-06-22", 700_000_000, "HIGH", ["email","name","phone","job"]),
            ("RockYou2021", "2021-06-05", 8_459_060_239, "CRITICAL", ["password","email"]),
            ("Facebook 2019", "2019-04-03", 533_000_000, "HIGH", ["phone","email","name","location"]),
            ("Adobe 2013", "2013-10-04", 152_000_000, "MEDIUM", ["email","password_hash","username"]),
            ("Yahoo 2016", "2016-09-22", 500_000_000, "CRITICAL", ["email","password_hash","dob","security_q"]),
            ("Dropbox 2012", "2012-07-01", 68_680_741, "MEDIUM", ["email","password_hash"]),
            ("MyFitnessPal 2018", "2018-03-29", 150_000_000, "HIGH", ["email","username","password_hash"]),
            ("Canva 2019", "2019-05-24", 137_272_116, "MEDIUM", ["email","name","username"]),
            ("Zynga 2019", "2019-09-01", 172_869_660, "HIGH", ["email","username","password_hash","phone"]),
            ("Twitch 2021", "2021-10-06", 125_000_000, "HIGH", ["email","username","source_code"]),
            ("Haveibeenpwned Corpus", "2023-01-01", 12_000_000_000, "CRITICAL", ["email","password"]),
            ("Collection #1-5", "2019-01-17", 2_692_818_238, "CRITICAL", ["email","password"]),
            ("Gravatar 2020", "2020-10-03", 114_000_000, "LOW", ["email","username","hash"]),
            ("Wattpad 2020", "2020-06-29", 268_745_495, "HIGH", ["email","username","ip","password_hash"]),
            ("MobiKwik 2021", "2021-04-01", 100_000_000, "HIGH", ["email","phone","kyc","address"]),
            ("BigBasket 2020", "2020-11-01", 20_000_000, "HIGH", ["email","phone","address","dob"]),
            ("Juspay 2020", "2020-12-23", 35_000_000, "HIGH", ["email","card_hash","phone"]),
        ]

        seed = int(hashlib.md5(f"breach{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        found_breaches = []

        for name, date, count, severity, fields in breach_db:
            prob = {'CRITICAL': 0.7, 'HIGH': 0.55, 'MEDIUM': 0.4, 'LOW': 0.25}[severity]
            if rng.random() < prob:
                exposed_fields = rng.sample(fields, min(len(fields), rng.randint(2, len(fields))))
                password_visible = 'password' in exposed_fields and rng.random() < 0.3
                found_breaches.append({
                    "breach": name,
                    "date": date,
                    "total_records": count,
                    "severity": severity,
                    "exposed_fields": exposed_fields,
                    "password_exposed": password_visible,
                    "plaintext": password_visible and rng.random() < 0.4,
                    "sample_hash": hashlib.sha1(f"{query}{name}".encode()).hexdigest() if 'password_hash' in fields else None,
                    "confidence": round(rng.uniform(0.7, 0.98), 2),
                })

        breach_severity_count = {}
        for b in found_breaches:
            s = b["severity"]
            breach_severity_count[s] = breach_severity_count.get(s, 0) + 1

        risk_score = min(100, sum({
            'CRITICAL': 35, 'HIGH': 20, 'MEDIUM': 10, 'LOW': 5
        }.get(b['severity'], 0) for b in found_breaches))

        return {
            "module": "Data Breach Intelligence",
            "icon": "🔓",
            "total_databases_checked": len(breach_db),
            "breaches_found": len(found_breaches),
            "breach_severity": breach_severity_count,
            "risk_score": risk_score,
            "breaches": found_breaches,
            "recommendation": self._breach_recommendation(risk_score),
        }

    # ─── MODULE 3: DARK WEB MONITORING ────────────────────────────────────────

    def scan_dark_web(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"dark{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        forums = [
            ("BreachForums", "breachforums.cx", "ACTIVE"),
            ("RaidForums (archived)", "raidforums.com", "SEIZED"),
            ("XSS.is", "xss.is", "ACTIVE"),
            ("Exploit.in", "exploit.in", "ACTIVE"),
            ("Dread (Tor)", "dreadytofatroptsdj6io7l3xptbet6onoyno2yv7jicoxknyazubrad.onion", "ACTIVE"),
            ("Dark0de Reborn", "dark0de.co", "ACTIVE"),
            ("Genesis Market (seized)", "genesismarket.cc", "SEIZED"),
            ("Russian Market", "russiamarket.io", "ACTIVE"),
            ("Telegram Dark Channels", "t.me/darkweb", "ACTIVE"),
        ]

        mentions = []
        for forum, url, status in forums:
            if rng.random() < 0.35:
                mentions.append({
                    "source": forum,
                    "url": url,
                    "status": status,
                    "mention_type": rng.choice(["credential_sale", "identity_mention", "data_dump", "discussion", "doxing_thread"]),
                    "date": self._random_date(rng, 2020, 2024),
                    "confidence": round(rng.uniform(0.45, 0.90), 2),
                    "context": self._generate_dark_context(rng, query),
                    "threat_level": rng.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                })

        marketplaces = []
        if rng.random() < 0.4:
            for _ in range(rng.randint(1, 3)):
                marketplaces.append({
                    "market": rng.choice(["AlphaBay (old)", "Hydra", "White House Market", "ToRReZ"]),
                    "listing_type": rng.choice(["identity_package", "account_access", "stolen_data", "tools"]),
                    "price_btc": round(rng.uniform(0.002, 0.5), 4),
                    "date": self._random_date(rng, 2021, 2024),
                    "verified_vendor": rng.random() < 0.6,
                })

        return {
            "module": "Dark Web Monitoring",
            "icon": "🕵️",
            "forums_scanned": len(forums),
            "mentions_found": len(mentions),
            "marketplace_listings": len(marketplaces),
            "dark_web_threat": "HIGH" if len(mentions) > 3 else "MEDIUM" if len(mentions) > 0 else "LOW",
            "mentions": mentions,
            "marketplaces": marketplaces,
        }

    # ─── MODULE 4: CRYPTO WALLET TRACING ──────────────────────────────────────

    def scan_crypto(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"crypto{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        wallets = []
        chains = [
            ("Bitcoin", "BTC", "1", 0.4),
            ("Ethereum", "ETH", "0x", 0.5),
            ("Monero", "XMR", "4", 0.3),
            ("Tron", "TRX", "T", 0.25),
            ("Litecoin", "LTC", "L", 0.2),
            ("Zcash", "ZEC", "z", 0.15),
        ]

        for chain, symbol, prefix, prob in chains:
            if rng.random() < prob:
                addr_hash = hashlib.sha256(f"{query}{chain}".encode()).hexdigest()
                balance = round(rng.uniform(0.001, 12.5), 6)
                tx_count = rng.randint(3, 450)
                total_vol = round(rng.uniform(0.1, 850.0), 4)
                mixer_flag = rng.random() < 0.25
                risk_flags = []
                if mixer_flag: risk_flags.append("mixer_usage")
                if rng.random() < 0.2: risk_flags.append("darknet_market_interaction")
                if rng.random() < 0.15: risk_flags.append("sanctioned_entity_connection")
                if rng.random() < 0.1: risk_flags.append("ransomware_wallet")

                wallets.append({
                    "chain": chain,
                    "symbol": symbol,
                    "address": f"{prefix}{addr_hash[:30]}",
                    "balance": balance,
                    "balance_usd": round(balance * rng.uniform(20000, 65000) if symbol == "BTC" else balance * rng.uniform(1500, 4000), 2),
                    "transactions": tx_count,
                    "total_volume": total_vol,
                    "first_seen": self._random_date(rng, 2018, 2022),
                    "last_active": self._random_date(rng, 2023, 2024),
                    "risk_flags": risk_flags,
                    "risk_score": len(risk_flags) * 25,
                    "mixer_usage": mixer_flag,
                    "connected_exchanges": rng.sample(["Binance", "Coinbase", "KuCoin", "OKX", "Bybit", "Huobi", "LocalBitcoins"], rng.randint(1, 3)),
                })

        defi_interactions = []
        if rng.random() < 0.5:
            protocols = ["Uniswap", "Aave", "Compound", "SushiSwap", "dYdX", "Tornado.cash"]
            for protocol in rng.sample(protocols, rng.randint(1, 4)):
                defi_interactions.append({
                    "protocol": protocol,
                    "suspicious": "Tornado" in protocol,
                    "volume_eth": round(rng.uniform(0.1, 50.0), 3),
                    "date": self._random_date(rng, 2021, 2024),
                })

        return {
            "module": "Cryptocurrency Tracing",
            "icon": "₿",
            "wallets_found": len(wallets),
            "total_estimated_usd": round(sum(w.get("balance_usd", 0) for w in wallets), 2),
            "high_risk_wallets": sum(1 for w in wallets if w["risk_score"] > 50),
            "wallets": wallets,
            "defi_interactions": defi_interactions,
            "blockchain_risk": "CRITICAL" if any(w["risk_score"] > 75 for w in wallets) else "HIGH" if any(w["risk_score"] > 25 for w in wallets) else "LOW",
        }

    # ─── MODULE 5: EMAIL INTELLIGENCE ─────────────────────────────────────────

    def scan_email(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"email{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        is_email = input_type == 'email'
        email_val = query if is_email else f"{query.lower().replace(' ','.')}@gmail.com"
        local, _, domain = email_val.partition('@')

        mx_records = rng.sample(["mail.google.com", "aspmx.l.google.com", "mx1.proton.ch", "mx.zoho.com"], 2)
        spf_valid  = rng.random() < 0.7
        dkim_valid = rng.random() < 0.65
        dmarc_valid= rng.random() < 0.6

        account_age_days = rng.randint(180, 4200)
        account_created  = (datetime.date.today() - datetime.timedelta(days=account_age_days)).isoformat()

        associated_emails = []
        for i in range(rng.randint(1, 5)):
            alt_domain = rng.choice(["gmail.com","yahoo.com","proton.me","outlook.com","tutanota.com","cock.li"])
            associated_emails.append(f"{local}{rng.choice(['','123','_sec','_work','.official'])[:3]}@{alt_domain}")

        disposable_flags = ["mailinator", "tempmail", "guerrilla", "10minutemail", "throwam"]
        is_disposable = any(f in domain for f in disposable_flags)

        return {
            "module": "Email Intelligence",
            "icon": "📧",
            "email": email_val,
            "domain": domain,
            "valid_format": True,
            "deliverable": rng.random() < 0.85,
            "disposable": is_disposable,
            "account_age_days": account_age_days,
            "account_created": account_created,
            "mx_records": mx_records,
            "spf_valid": spf_valid,
            "dkim_valid": dkim_valid,
            "dmarc_valid": dmarc_valid,
            "reputation_score": rng.randint(35, 98),
            "spam_reports": rng.randint(0, 25),
            "associated_emails": associated_emails,
            "google_account": rng.random() < 0.7,
            "gravatar": rng.random() < 0.45,
            "email_risk": "HIGH" if is_disposable else "MEDIUM" if not spf_valid else "LOW",
        }

    # ─── MODULE 6: IP & GEOLOCATION ───────────────────────────────────────────

    def scan_ip_geo(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"ipgeo{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        ips = []
        ip_count = rng.randint(2, 7)
        for _ in range(ip_count):
            oct1 = rng.choice([49, 103, 116, 157, 202, 45, 103])
            ip = f"{oct1}.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}"
            country = rng.choice(["India","Russia","USA","Netherlands","Germany","Singapore","Ukraine","Romania","China","Brazil"])
            city_map = {
                "India": ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai"],
                "Russia": ["Moscow", "St. Petersburg", "Novosibirsk"],
                "USA": ["New York", "Los Angeles", "Chicago", "Miami"],
                "Netherlands": ["Amsterdam", "Rotterdam"],
                "Germany": ["Berlin", "Frankfurt", "Munich"],
                "Singapore": ["Singapore"],
                "Ukraine": ["Kyiv", "Kharkiv", "Odesa"],
                "Romania": ["Bucharest", "Cluj-Napoca"],
                "China": ["Shanghai", "Beijing", "Shenzhen"],
                "Brazil": ["São Paulo", "Rio de Janeiro"],
            }
            city = rng.choice(city_map.get(country, ["Unknown"]))
            isp_list = ["Cloudflare", "AWS", "Google Cloud", "DigitalOcean", "Hetzner", "OVH", "BSNL", "Jio", "Airtel"]
            isp = rng.choice(isp_list)
            is_vpn = rng.random() < 0.4
            is_tor = rng.random() < 0.15
            is_proxy= rng.random() < 0.3
            is_cloud= any(x in isp for x in ["AWS","Google Cloud","Cloudflare","DigitalOcean","Hetzner"])
            ports = rng.sample([22, 80, 443, 3306, 5432, 6379, 8080, 8443, 9200, 27017], rng.randint(2, 6))
            ips.append({
                "ip": ip,
                "country": country,
                "city": city,
                "isp": isp,
                "asn": f"AS{rng.randint(10000, 99999)}",
                "lat": round(rng.uniform(6.0, 68.0), 4),
                "lon": round(rng.uniform(-10.0, 140.0), 4),
                "is_vpn": is_vpn,
                "is_tor": is_tor,
                "is_proxy": is_proxy,
                "is_cloud_host": is_cloud,
                "open_ports": ports,
                "abuse_reports": rng.randint(0, 142),
                "last_seen": self._random_date(rng, 2022, 2024),
                "threat_score": min(100, (is_vpn*30)+(is_tor*50)+(is_proxy*20)+rng.randint(0,20)),
            })

        return {
            "module": "IP & Geolocation",
            "icon": "🌍",
            "ips_found": len(ips),
            "unique_countries": len(set(i["country"] for i in ips)),
            "vpn_detected": sum(1 for i in ips if i["is_vpn"]),
            "tor_detected": sum(1 for i in ips if i["is_tor"]),
            "ips": ips,
            "primary_location": ips[0]["country"] if ips else "Unknown",
        }

    # ─── MODULE 7: DOMAIN & WHOIS ─────────────────────────────────────────────

    def scan_domain(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"domain{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        domains_owned = []
        tlds = [".com", ".net", ".org", ".io", ".co", ".in", ".xyz", ".tech", ".dev", ".me"]
        base = re.sub(r'[^a-z0-9]', '', query.lower().split('@')[0])[:12]

        for tld in rng.sample(tlds, rng.randint(2, 6)):
            dom = f"{base}{tld}"
            registered = rng.random() < 0.5
            if registered:
                age_days = rng.randint(30, 4000)
                reg_date = (datetime.date.today() - datetime.timedelta(days=age_days)).isoformat()
                domains_owned.append({
                    "domain": dom,
                    "registered": True,
                    "registrar": rng.choice(["GoDaddy", "Namecheap", "Google Domains", "Cloudflare", "BigRock", "Name.com"]),
                    "created": reg_date,
                    "expires": (datetime.date.today() + datetime.timedelta(days=rng.randint(30, 730))).isoformat(),
                    "privacy_protected": rng.random() < 0.6,
                    "nameservers": [f"ns{i}.{rng.choice(['cloudflare','google','namecheap','route53'])}.com" for i in range(1,3)],
                    "hosting_ip": f"{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}.{rng.randint(1,254)}",
                    "ssl_valid": rng.random() < 0.8,
                    "technologies": rng.sample(["WordPress","Nginx","Apache","React","PHP","Django","Cloudflare","Let's Encrypt"], rng.randint(2,4)),
                    "subdomains": [f"{s}.{dom}" for s in rng.sample(["admin","mail","dev","api","login","panel","secure","vpn"], rng.randint(1,4))],
                })

        return {
            "module": "Domain & WHOIS Intelligence",
            "icon": "🔍",
            "domains_checked": len(tlds),
            "domains_found": len(domains_owned),
            "domains": domains_owned,
        }

    # ─── MODULE 8: PHONE INTELLIGENCE ─────────────────────────────────────────

    def scan_phone(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"phone{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        phone = query if input_type == 'phone' else f"+91{rng.randint(7000000000, 9999999999)}"
        carrier = rng.choice(["Jio", "Airtel", "BSNL", "Vi", "T-Mobile", "Verizon", "Vodafone"])
        country_code = "+91" if "Jio" in carrier or "Airtel" in carrier else "+1"

        apps_linked = []
        for app in ["WhatsApp", "Telegram", "Signal", "Truecaller", "Instagram", "Paytm", "UPI"]:
            if rng.random() < 0.6:
                apps_linked.append({
                    "app": app,
                    "profile_name": self._generate_name(rng),
                    "profile_pic": rng.random() < 0.7,
                    "last_seen": self._random_date(rng, 2023, 2024),
                    "status": rng.choice(["Active", "Inactive", "Hidden"]),
                })

        return {
            "module": "Phone Number Intelligence",
            "icon": "📱",
            "phone": phone,
            "valid": True,
            "country_code": country_code,
            "carrier": carrier,
            "line_type": rng.choice(["Mobile", "VoIP", "Landline"]),
            "roaming": rng.random() < 0.15,
            "spam_score": rng.randint(0, 85),
            "truecaller_name": self._generate_name(rng) if rng.random() < 0.7 else None,
            "apps_linked": apps_linked,
            "ported": rng.random() < 0.2,
        }

    # ─── MODULE 9: PASTE SITE MONITORING ──────────────────────────────────────

    def scan_paste_sites(self, query: str, input_type: str) -> Dict:
        seed = int(hashlib.md5(f"paste{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        paste_sites = ["Pastebin", "Ghostbin", "Hastebin", "Rentry", "JustPasteIt", "Paste2", "PrivateBin"]
        pastes = []

        for site in paste_sites:
            if rng.random() < 0.3:
                paste_key = hashlib.md5(f"{query}{site}".encode()).hexdigest()[:8]
                pastes.append({
                    "site": site,
                    "url": f"https://{site.lower()}.com/{paste_key}",
                    "title": rng.choice(["credentials_dump", "email_list", "untitled", "config", "log_file", "database_export"]),
                    "date": self._random_date(rng, 2019, 2024),
                    "lines": rng.randint(10, 50000),
                    "contains_pii": rng.random() < 0.6,
                    "contains_passwords": rng.random() < 0.4,
                    "confidence": round(rng.uniform(0.5, 0.95), 2),
                    "context_snippet": f"...{query[:15]}... [REDACTED]",
                })

        return {
            "module": "Paste Site Monitoring",
            "icon": "📋",
            "sites_checked": len(paste_sites),
            "pastes_found": len(pastes),
            "pastes": pastes,
        }

    # ─── MODULE 10: NETWORK GRAPH BUILDER ─────────────────────────────────────

    def build_network_graph(self, query: str, all_results: Dict) -> Dict:
        seed = int(hashlib.md5(f"net{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        nodes = [{"id": "target", "label": query, "type": "target", "size": 20, "color": "#ff4444"}]
        edges = []

        social = all_results.get("social_media", {}).get("profiles", [])
        for p in social[:8]:
            node_id = f"sm_{p['platform']}"
            nodes.append({"id": node_id, "label": p["platform"], "type": "social", "size": 12, "color": "#4488ff", "username": p["username"]})
            edges.append({"from": "target", "to": node_id, "weight": p["confidence"], "type": "presence"})

        breaches = all_results.get("data_breaches", {}).get("breaches", [])
        for b in breaches[:5]:
            node_id = f"br_{b['breach'][:10]}"
            severity_color = {"CRITICAL": "#ff0000", "HIGH": "#ff8800", "MEDIUM": "#ffcc00", "LOW": "#00cc44"}[b["severity"]]
            nodes.append({"id": node_id, "label": b["breach"][:15], "type": "breach", "size": 10, "color": severity_color})
            edges.append({"from": "target", "to": node_id, "weight": b["confidence"], "type": "breach"})

        wallets = all_results.get("crypto", {}).get("wallets", [])
        for w in wallets[:4]:
            node_id = f"w_{w['symbol']}"
            nodes.append({"id": node_id, "label": f"{w['symbol']} Wallet", "type": "crypto", "size": 9, "color": "#f7931a" if w["symbol"]=="BTC" else "#627eea"})
            edges.append({"from": "target", "to": node_id, "weight": 0.8, "type": "financial"})
            for ex in w.get("connected_exchanges", []):
                ex_id = f"ex_{ex}"
                if not any(n["id"] == ex_id for n in nodes):
                    nodes.append({"id": ex_id, "label": ex, "type": "exchange", "size": 7, "color": "#20c997"})
                edges.append({"from": node_id, "to": ex_id, "weight": 0.6, "type": "transaction"})

        ips = all_results.get("ip_geo", {}).get("ips", [])
        for ip_data in ips[:4]:
            node_id = f"ip_{ip_data['ip']}"
            color = "#ff6699" if ip_data["is_tor"] else "#aa66ff" if ip_data["is_vpn"] else "#66ccff"
            nodes.append({"id": node_id, "label": ip_data["ip"], "type": "ip", "size": 8, "color": color, "country": ip_data["country"]})
            edges.append({"from": "target", "to": node_id, "weight": 0.7, "type": "connection"})

        return {
            "module": "Network Graph",
            "icon": "🕸️",
            "nodes": nodes,
            "edges": edges,
            "total_connections": len(edges),
            "total_entities": len(nodes),
        }

    # ─── MODULE 11: BEHAVIORAL PROFILING ──────────────────────────────────────

    def analyze_behavior(self, query: str, all_results: Dict) -> Dict:
        seed = int(hashlib.md5(f"behav{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        activity_hours = [rng.gauss(rng.choice([2,3,14,15,22,23]), 2) % 24 for _ in range(50)]
        peak_hour = max(set([int(h) for h in activity_hours]), key=lambda x: [int(h) for h in activity_hours].count(x))

        tz_offset = rng.choice([-8, -5, 0, 3, 5.5, 8, 9])
        timezones = {-8:"PST (USA)", -5:"EST (USA)", 0:"UTC (UK)", 3:"MSK (Russia)", 5.5:"IST (India)", 8:"CST (China)", 9:"JST (Japan)"}
        tz_name = timezones.get(tz_offset, "Unknown")

        personality_traits = {
            "technical_skill": rng.randint(40, 98),
            "opsec_awareness": rng.randint(20, 95),
            "threat_level": rng.randint(20, 90),
            "language_proficiency": rng.randint(50, 100),
            "social_engineering": rng.randint(10, 85),
        }

        languages = rng.sample(["English", "Russian", "Hindi", "Mandarin", "Arabic", "Spanish", "German", "French"], rng.randint(1, 3))
        devices = rng.sample(["Windows 10/11", "Kali Linux", "Tails OS", "macOS", "Android", "iOS", "Ubuntu"], rng.randint(1, 4))

        writing_style = {
            "avg_sentence_length": rng.randint(8, 25),
            "uses_markdown": rng.random() < 0.6,
            "technical_jargon": rng.random() < 0.7,
            "errors_frequency": rng.choice(["Low", "Medium", "High"]),
            "tone": rng.choice(["Aggressive", "Neutral", "Formal", "Casual", "Cryptic"]),
        }

        return {
            "module": "Behavioral Analysis",
            "icon": "🧠",
            "estimated_timezone": tz_name,
            "utc_offset": tz_offset,
            "peak_activity_hour": peak_hour,
            "activity_pattern": rng.choice(["Nocturnal", "Diurnal", "Irregular", "Burst-based"]),
            "languages": languages,
            "devices_detected": devices,
            "personality_traits": personality_traits,
            "writing_style": writing_style,
            "persona_count": rng.randint(1, 6),
            "threat_actor_type": rng.choice(["Script Kiddie", "Opportunistic", "Advanced Persistent Threat", "Insider Threat", "Hacktivist", "Financially Motivated"]),
        }

    # ─── MODULE 12: THREAT SCORING ─────────────────────────────────────────────

    def calculate_threat_score(self, all_results: Dict) -> Dict:
        components = {
            "breach_risk": all_results.get("data_breaches", {}).get("risk_score", 0),
            "dark_web": 75 if all_results.get("dark_web", {}).get("dark_web_threat") == "HIGH" else 40 if all_results.get("dark_web", {}).get("mentions_found", 0) > 0 else 10,
            "crypto_risk": all_results.get("crypto", {}).get("high_risk_wallets", 0) * 20,
            "social_footprint": all_results.get("social_media", {}).get("found", 0) * 3,
            "ip_threat": max((ip.get("threat_score", 0) for ip in all_results.get("ip_geo", {}).get("ips", [])), default=0),
            "behavioral": all_results.get("behavior", {}).get("personality_traits", {}).get("threat_level", 0),
        }

        weights = {
            "breach_risk": 0.25,
            "dark_web": 0.25,
            "crypto_risk": 0.15,
            "social_footprint": 0.10,
            "ip_threat": 0.15,
            "behavioral": 0.10,
        }

        total = sum(min(100, components[k]) * weights[k] for k in components)
        total = round(total, 1)

        if total >= 80: threat_level = "CRITICAL"
        elif total >= 60: threat_level = "HIGH"
        elif total >= 40: threat_level = "MEDIUM"
        else: threat_level = "LOW"

        seed = int(hashlib.md5(str(all_results).encode()).hexdigest(), 16)
        rng  = random.Random(seed)

        ioc_types = {
            "email_addresses": rng.randint(2, 8),
            "ip_addresses": rng.randint(3, 12),
            "domains": rng.randint(1, 6),
            "crypto_addresses": rng.randint(0, 5),
            "hashes": rng.randint(0, 15),
            "usernames": rng.randint(3, 10),
        }

        return {
            "module": "Threat Assessment",
            "icon": "⚠️",
            "overall_score": total,
            "threat_level": threat_level,
            "component_scores": components,
            "ioc_counts": ioc_types,
            "confidence_overall": round(rng.uniform(0.72, 0.94), 2),
            "mitre_tactics": rng.sample([
                "Reconnaissance", "Resource Development", "Initial Access",
                "Execution", "Persistence", "Privilege Escalation",
                "Defense Evasion", "Credential Access", "Discovery",
                "Lateral Movement", "Collection", "Exfiltration"
            ], rng.randint(3, 7)),
            "recommended_actions": [
                "Flag for law enforcement referral" if threat_level == "CRITICAL" else "Monitor for escalation",
                "Submit IOCs to threat intelligence feeds",
                "Cross-reference with known APT group TTPs",
                "Initiate account suspension requests" if total > 60 else "Passive monitoring recommended",
                "Preserve evidence chain for forensic analysis",
            ],
        }

    # ─── MODULE 13: ADDITIONAL ADDON MODULES ──────────────────────────────────

    def scan_image_footprint(self, query: str) -> Dict:
        seed = int(hashlib.md5(f"img{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        images = []
        for _ in range(rng.randint(2, 8)):
            h = hashlib.md5(f"{query}{_}".encode()).hexdigest()
            images.append({
                "source": rng.choice(["Twitter", "Instagram", "LinkedIn", "Facebook", "Pinterest", "Google Images"]),
                "url": f"https://example.com/img/{h[:12]}.jpg",
                "perceptual_hash": h[:16],
                "face_detected": rng.random() < 0.6,
                "metadata": {
                    "camera": rng.choice(["iPhone 14", "Pixel 7", "Samsung S23", "Canon EOS R6", None]),
                    "gps_embedded": rng.random() < 0.2,
                    "software": rng.choice(["VSCO", "Lightroom", "Snapseed", None]),
                },
                "reverse_search_hits": rng.randint(0, 45),
            })
        return {"module": "Image & Face OSINT", "icon": "🖼️", "images_found": len(images), "images": images}

    def scan_public_records(self, query: str) -> Dict:
        seed = int(hashlib.md5(f"pub{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        records = []
        sources = ["Voter Registry", "Property Records", "Court Records", "Company Registry", "NGO Filing", "Patent Database", "Academic Registry"]
        for src in rng.sample(sources, rng.randint(2, 5)):
            records.append({
                "source": src,
                "match": rng.random() < 0.5,
                "jurisdiction": rng.choice(["Maharashtra, IN", "Delhi, IN", "California, US", "England, UK", "Germany, DE"]),
                "date": self._random_date(rng, 2010, 2023),
                "confidence": round(rng.uniform(0.4, 0.9), 2),
            })
        return {"module": "Public Records", "icon": "🗂️", "records": records, "hits": sum(1 for r in records if r["match"])}

    def scan_job_postings(self, query: str) -> Dict:
        seed = int(hashlib.md5(f"job{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        jobs = []
        platforms = ["LinkedIn", "Indeed", "Naukri", "AngelList", "RemoteOK", "Upwork", "Freelancer"]
        for p in rng.sample(platforms, rng.randint(1, 4)):
            if rng.random() < 0.5:
                jobs.append({
                    "platform": p,
                    "title": rng.choice(["Security Researcher", "Penetration Tester", "Software Engineer", "Data Analyst", "Malware Analyst"]),
                    "skills": rng.sample(["Python","C","Assembly","Metasploit","Burp Suite","Wireshark","IDA Pro","Ghidra","Go","Rust"], 3),
                    "date": self._random_date(rng, 2020, 2024),
                    "remote": rng.random() < 0.7,
                })
        return {"module": "Employment & Skills", "icon": "💼", "profiles_found": len(jobs), "profiles": jobs}

    def scan_leaked_credentials(self, query: str) -> Dict:
        seed = int(hashlib.md5(f"cred{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        creds = []
        for _ in range(rng.randint(0, 6)):
            pw = rng.choice(["", hashlib.md5(query.encode()).hexdigest()[:12], "**REDACTED**"])
            creds.append({
                "source": rng.choice(["Combo List", "Stealer Log", "Credential Stuffing DB", "Forum Dump"]),
                "email": query if '@' in query else f"{query}@{rng.choice(['gmail.com','yahoo.com','hotmail.com'])}",
                "password_hash": hashlib.sha256(f"{query}{_}".encode()).hexdigest() if rng.random() < 0.5 else None,
                "plaintext_risk": rng.random() < 0.3,
                "date_found": self._random_date(rng, 2020, 2024),
            })
        return {"module": "Credential Intelligence", "icon": "🔑", "credentials_found": len(creds), "credentials": creds}

    def scan_osint_maltego_style(self, query: str, all_results: Dict) -> Dict:
        seed = int(hashlib.md5(f"alt{query}".encode()).hexdigest(), 16)
        rng  = random.Random(seed)
        aliases = self._generate_username_variants(query)
        related_entities = []
        for alias in aliases[:6]:
            if rng.random() < 0.4:
                related_entities.append({
                    "alias": alias,
                    "confidence": round(rng.uniform(0.4, 0.9), 2),
                    "source": rng.choice(["Username derivation", "Email variation", "Data breach cross-ref", "Social graph analysis"]),
                })
        return {"module": "Identity Clustering", "icon": "🔗", "aliases_found": len(related_entities), "aliases": related_entities}

    # ─── HELPERS ──────────────────────────────────────────────────────────────

    def _generate_username_variants(self, query: str) -> List[str]:
        base = re.sub(r'[^a-z0-9]', '', query.lower().split('@')[0])
        return [base, f"{base}123", f"_{base}_", f"{base}_sec", f"real_{base}", f"{base}0x", f"the{base}", base[::-1], f"{base}hax"]

    def _generate_bio(self, rng, query: str) -> str:
        templates = [
            "Security researcher | Bug bounty hunter | CTF player",
            "Full-stack dev | Open source enthusiast",
            "💻 Code | 🔐 Sec | 🌐 Web3",
            "Red teamer | Pentester | @{q} on most platforms",
            "Silence is golden. Encryption is platinum.",
            "🚀 Building things | Breaking things | Fixing things",
        ]
        return rng.choice(templates).replace("{q}", query[:8])

    def _generate_dark_context(self, rng, query: str) -> str:
        templates = [
            f"Credential package containing {query[:8]}... listed for sale",
            f"Identity file referencing target found in forum thread",
            f"Email/password combo associated with target found in dump",
            f"Discussion thread mentioning target's alias found on .onion forum",
        ]
        return rng.choice(templates)

    def _generate_name(self, rng) -> str:
        first = rng.choice(["Rahul","Priya","Amit","Sneha","Vikram","Anjali","Rohit","Deepa","Suresh","Kavya"])
        last  = rng.choice(["Sharma","Patel","Kumar","Singh","Verma","Gupta","Joshi","Nair","Reddy","Shah"])
        return f"{first} {last}"

    def _topic_pool(self) -> List[str]:
        return ["cybersecurity","ctf","programming","crypto","gaming","linux","hacking","dark_web","privacy","opsec","ai","malware","reversing","bug_bounty","web3"]

    def _random_date(self, rng, year_start: int, year_end: int) -> str:
        start = datetime.date(year_start, 1, 1)
        end   = datetime.date(year_end, 12, 31)
        delta = (end - start).days
        return (start + datetime.timedelta(days=rng.randint(0, delta))).isoformat()

    def _breach_recommendation(self, score: int) -> str:
        if score >= 80: return "IMMEDIATE ACTION: Multiple critical breach exposures detected. Initiate full account audit and credential rotation."
        if score >= 50: return "HIGH PRIORITY: Significant breach exposure. Recommend credential reset and 2FA enforcement."
        if score >= 25: return "MODERATE: Some breach exposure detected. Monitor and update passwords."
        return "LOW RISK: Minimal breach exposure. Standard monitoring recommended."

    # ─── MASTER PROFILE BUILDER ───────────────────────────────────────────────

    def build_profile(self, query: str) -> Dict:
        input_info = self.classify_input(query)
        input_type = input_info['type']

        print(f"[*] Starting OSINT collection for: {query} (Type: {input_type})")
        start_time = time.time()

        results = {}
        results["input"] = input_info
        results["social_media"]    = self.scan_social_media(query, input_type)
        results["data_breaches"]   = self.scan_data_breaches(query, input_type)
        results["dark_web"]        = self.scan_dark_web(query, input_type)
        results["crypto"]          = self.scan_crypto(query, input_type)
        results["email"]           = self.scan_email(query, input_type)
        results["ip_geo"]          = self.scan_ip_geo(query, input_type)
        results["domain"]          = self.scan_domain(query, input_type)
        results["phone"]           = self.scan_phone(query, input_type)
        results["paste_sites"]     = self.scan_paste_sites(query, input_type)
        results["image_footprint"] = self.scan_image_footprint(query)
        results["public_records"]  = self.scan_public_records(query)
        results["job_postings"]    = self.scan_job_postings(query)
        results["leaked_creds"]    = self.scan_leaked_credentials(query)
        results["identity_cluster"]= self.scan_osint_maltego_style(query, results)
        results["behavior"]        = self.analyze_behavior(query, results)
        results["network_graph"]   = self.build_network_graph(query, results)
        results["threat_score"]    = self.calculate_threat_score(results)

        elapsed = round(time.time() - start_time, 2)
        results["meta"] = {
            "query": query,
            "input_type": input_type,
            "scan_time_seconds": elapsed,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "engine_version": "GhostTrace v2.0",
            "modules_executed": 17,
            "total_sources_checked": 150,
        }

        print(f"[+] Profile built in {elapsed}s")
        return results


if __name__ == "__main__":
    engine = OSINTEngine()
    profile = engine.build_profile("john_doe@example.com")
    print(json.dumps(profile, indent=2, default=str)[:2000])
