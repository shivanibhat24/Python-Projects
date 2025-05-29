#!/usr/bin/env python3
"""
Phishing-Aware Email Agent using Model Context Protocol
Analyzes emails for phishing indicators and learns patterns over time.
"""

import re
import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhishingPattern:
    """Represents a suspicious pattern detected in emails"""
    pattern_type: str
    pattern_value: str
    confidence_score: float
    first_seen: str
    last_seen: str
    occurrence_count: int
    context: str

@dataclass
class EmailAnalysisResult:
    """Results of email phishing analysis"""
    email_id: str
    risk_score: float
    detected_patterns: List[PhishingPattern]
    recommendations: List[str]
    is_suspicious: bool
    analysis_timestamp: str

class ModelContextProtocol:
    """Simple MCP implementation for pattern memory"""
    
    def __init__(self, db_path: str = "phishing_patterns.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for pattern storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_value TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL,
                context TEXT,
                UNIQUE(pattern_type, pattern_value)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_analyses (
                email_id TEXT PRIMARY KEY,
                risk_score REAL NOT NULL,
                detected_patterns TEXT,
                recommendations TEXT,
                is_suspicious BOOLEAN NOT NULL,
                analysis_timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_pattern(self, pattern: PhishingPattern):
        """Store or update a phishing pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO patterns 
                (pattern_type, pattern_value, confidence_score, first_seen, last_seen, occurrence_count, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_type,
                pattern.pattern_value,
                pattern.confidence_score,
                pattern.first_seen,
                pattern.last_seen,
                pattern.occurrence_count,
                pattern.context
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
        finally:
            conn.close()
    
    def get_patterns(self, pattern_type: Optional[str] = None) -> List[PhishingPattern]:
        """Retrieve stored patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if pattern_type:
            cursor.execute('SELECT * FROM patterns WHERE pattern_type = ?', (pattern_type,))
        else:
            cursor.execute('SELECT * FROM patterns')
        
        rows = cursor.fetchall()
        conn.close()
        
        patterns = []
        for row in rows:
            patterns.append(PhishingPattern(
                pattern_type=row[1],
                pattern_value=row[2],
                confidence_score=row[3],
                first_seen=row[4],
                last_seen=row[5],
                occurrence_count=row[6],
                context=row[7]
            ))
        
        return patterns
    
    def update_pattern_occurrence(self, pattern_type: str, pattern_value: str):
        """Update pattern occurrence count and last seen timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE patterns 
            SET occurrence_count = occurrence_count + 1,
                last_seen = ?
            WHERE pattern_type = ? AND pattern_value = ?
        ''', (datetime.now().isoformat(), pattern_type, pattern_value))
        
        conn.commit()
        conn.close()
    
    def store_analysis(self, result: EmailAnalysisResult):
        """Store email analysis result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO email_analyses
            (email_id, risk_score, detected_patterns, recommendations, is_suspicious, analysis_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.email_id,
            result.risk_score,
            json.dumps([asdict(p) for p in result.detected_patterns]),
            json.dumps(result.recommendations),
            result.is_suspicious,
            result.analysis_timestamp
        ))
        
        conn.commit()
        conn.close()

class PhishingDetector:
    """Core phishing detection engine"""
    
    def __init__(self):
        self.suspicious_domains = [
            'bit.ly', 'tinyurl.com', 'short.link', 'goo.gl',
            'paypal-security.com', 'amazon-security.net'
        ]
        
        self.urgent_keywords = [
            'urgent', 'immediate action', 'verify now', 'click here',
            'suspended', 'limited time', 'expires today', 'act now'
        ]
        
        self.spoofing_indicators = [
            'noreply', 'no-reply', 'automated', 'system',
            'security-team', 'account-verification'
        ]
    
    def analyze_urls(self, text: str) -> List[Tuple[str, float]]:
        """Analyze URLs in email text for suspicious patterns"""
        suspicious_urls = []
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s<>"\']*)?'
        
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        for url in urls:
            risk_score = 0.0
            
            # Check for URL shorteners
            if any(domain in url.lower() for domain in ['bit.ly', 'tinyurl.com', 'short.link']):
                risk_score += 0.7
            
            # Check for suspicious domains
            if any(domain in url.lower() for domain in self.suspicious_domains):
                risk_score += 0.8
            
            # Check for IP addresses instead of domains
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            if re.search(ip_pattern, url):
                risk_score += 0.9
            
            # Check for suspicious subdomains
            subdomain_pattern = r'[a-zA-Z0-9-]+\.(paypal|amazon|microsoft|google|apple)\.[a-zA-Z]+'
            if re.search(subdomain_pattern, url, re.IGNORECASE):
                risk_score += 0.85
            
            if risk_score > 0.3:
                suspicious_urls.append((url, min(risk_score, 1.0)))
        
        return suspicious_urls
    
    def analyze_content(self, text: str) -> List[Tuple[str, str, float]]:
        """Analyze email content for phishing indicators"""
        indicators = []
        text_lower = text.lower()
        
        # Check for urgent language
        for keyword in self.urgent_keywords:
            if keyword.lower() in text_lower:
                indicators.append(('urgent_language', keyword, 0.6))
        
        # Check for credential harvesting attempts
        credential_patterns = [
            r'verify.{0,20}(password|account|identity)',
            r'confirm.{0,20}(login|credentials|details)',
            r'update.{0,20}(payment|billing|card)'
        ]
        
        for pattern in credential_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(('credential_harvesting', match, 0.8))
        
        # Check for social engineering tactics
        social_patterns = [
            r'congratulations.{0,50}(won|winner|selected)',
            r'you.{0,10}have.{0,10}been.{0,10}chosen',
            r'claim.{0,20}(prize|reward|bonus)'
        ]
        
        for pattern in social_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                indicators.append(('social_engineering', match, 0.7))
        
        return indicators
    
    def analyze_sender(self, sender: str, reply_to: str = '') -> List[Tuple[str, str, float]]:
        """Analyze sender information for spoofing indicators"""
        indicators = []
        
        # Check for spoofing indicators in sender
        for indicator in self.spoofing_indicators:
            if indicator in sender.lower():
                indicators.append(('sender_spoofing', indicator, 0.6))
        
        # Check for mismatch between sender and reply-to
        if reply_to and sender != reply_to:
            if not any(domain in reply_to.lower() for domain in ['noreply', 'no-reply']):
                indicators.append(('sender_mismatch', f"{sender} vs {reply_to}", 0.7))
        
        return indicators

class PhishingAwareEmailAgent:
    """Main email agent with phishing detection and learning capabilities"""
    
    def __init__(self, db_path: str = "phishing_patterns.db"):
        self.mcp = ModelContextProtocol(db_path)
        self.detector = PhishingDetector()
        self.risk_threshold = 0.5
    
    def analyze_email(self, email_data: Dict[str, Any]) -> EmailAnalysisResult:
        """Analyze an email for phishing indicators"""
        email_id = self._generate_email_id(email_data)
        detected_patterns = []
        risk_score = 0.0
        recommendations = []
        
        # Extract email components
        sender = email_data.get('sender', '')
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        reply_to = email_data.get('reply_to', '')
        
        # Analyze URLs
        suspicious_urls = self.detector.analyze_urls(body)
        for url, score in suspicious_urls:
            pattern = PhishingPattern(
                pattern_type='suspicious_url',
                pattern_value=url,
                confidence_score=score,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                occurrence_count=1,
                context=f"Found in email from {sender}"
            )
            detected_patterns.append(pattern)
            risk_score += score * 0.3
        
        # Analyze content
        content_indicators = self.detector.analyze_content(subject + ' ' + body)
        for indicator_type, pattern_value, score in content_indicators:
            pattern = PhishingPattern(
                pattern_type=indicator_type,
                pattern_value=pattern_value,
                confidence_score=score,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                occurrence_count=1,
                context=f"Found in {indicator_type}"
            )
            detected_patterns.append(pattern)
            risk_score += score * 0.25
        
        # Analyze sender
        sender_indicators = self.detector.analyze_sender(sender, reply_to)
        for indicator_type, pattern_value, score in sender_indicators:
            pattern = PhishingPattern(
                pattern_type=indicator_type,
                pattern_value=pattern_value,
                confidence_score=score,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
                occurrence_count=1,
                context=f"Sender analysis: {sender}"
            )
            detected_patterns.append(pattern)
            risk_score += score * 0.2
        
        # Check against known patterns
        self._check_known_patterns(detected_patterns)
        
        # Normalize risk score
        risk_score = min(risk_score, 1.0)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(detected_patterns, risk_score)
        
        # Store patterns for learning
        for pattern in detected_patterns:
            self._store_or_update_pattern(pattern)
        
        # Create analysis result
        result = EmailAnalysisResult(
            email_id=email_id,
            risk_score=risk_score,
            detected_patterns=detected_patterns,
            recommendations=recommendations,
            is_suspicious=risk_score > self.risk_threshold,
            analysis_timestamp=datetime.now().isoformat()
        )
        
        # Store analysis result
        self.mcp.store_analysis(result)
        
        return result
    
    def _generate_email_id(self, email_data: Dict[str, Any]) -> str:
        """Generate unique ID for email"""
        content = f"{email_data.get('sender', '')}{email_data.get('subject', '')}{email_data.get('timestamp', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_known_patterns(self, current_patterns: List[PhishingPattern]):
        """Check current patterns against known patterns and update confidence"""
        known_patterns = self.mcp.get_patterns()
        
        for current in current_patterns:
            for known in known_patterns:
                if (current.pattern_type == known.pattern_type and 
                    current.pattern_value.lower() in known.pattern_value.lower()):
                    current.confidence_score = min(current.confidence_score + 0.1, 1.0)
                    self.mcp.update_pattern_occurrence(known.pattern_type, known.pattern_value)
    
    def _store_or_update_pattern(self, pattern: PhishingPattern):
        """Store new pattern or update existing one"""
        existing_patterns = self.mcp.get_patterns(pattern.pattern_type)
        
        pattern_exists = False
        for existing in existing_patterns:
            if existing.pattern_value == pattern.pattern_value:
                pattern_exists = True
                # Update existing pattern
                existing.occurrence_count += 1
                existing.last_seen = datetime.now().isoformat()
                existing.confidence_score = min(existing.confidence_score + 0.05, 1.0)
                self.mcp.store_pattern(existing)
                break
        
        if not pattern_exists:
            self.mcp.store_pattern(pattern)
    
    def _generate_recommendations(self, patterns: List[PhishingPattern], risk_score: float) -> List[str]:
        """Generate security recommendations based on detected patterns"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.append("HIGH RISK: Do not interact with this email. Mark as spam and delete.")
        elif risk_score > 0.6:
            recommendations.append("MEDIUM RISK: Exercise extreme caution. Verify sender through alternative means.")
        elif risk_score > 0.3:
            recommendations.append("LOW RISK: Be cautious and verify any links or attachments before clicking.")
        
        pattern_types = set(p.pattern_type for p in patterns)
        
        if 'suspicious_url' in pattern_types:
            recommendations.append("Suspicious URLs detected. Hover over links to verify destinations before clicking.")
        
        if 'credential_harvesting' in pattern_types:
            recommendations.append("Potential credential harvesting attempt. Never enter passwords via email links.")
        
        if 'sender_spoofing' in pattern_types:
            recommendations.append("Sender may be spoofed. Verify sender identity through official channels.")
        
        if 'urgent_language' in pattern_types:
            recommendations.append("Urgent language detected. Legitimate companies rarely require immediate action.")
        
        return recommendations
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        patterns = self.mcp.get_patterns()
        
        stats = {
            'total_patterns': len(patterns),
            'pattern_types': {},
            'high_confidence_patterns': 0,
            'most_common_patterns': []
        }
        
        for pattern in patterns:
            if pattern.pattern_type not in stats['pattern_types']:
                stats['pattern_types'][pattern.pattern_type] = 0
            stats['pattern_types'][pattern.pattern_type] += 1
            
            if pattern.confidence_score > 0.8:
                stats['high_confidence_patterns'] += 1
        
        # Get most common patterns
        sorted_patterns = sorted(patterns, key=lambda x: x.occurrence_count, reverse=True)
        stats['most_common_patterns'] = [
            {
                'type': p.pattern_type,
                'value': p.pattern_value[:50] + '...' if len(p.pattern_value) > 50 else p.pattern_value,
                'count': p.occurrence_count,
                'confidence': p.confidence_score
            }
            for p in sorted_patterns[:10]
        ]
        
        return stats

# Example usage and testing
def main():
    """Example usage of the Phishing-Aware Email Agent"""
    agent = PhishingAwareEmailAgent()
    
    # Example phishing emails for testing
    test_emails = [
        {
            'sender': 'security-team@paypal-security.com',
            'subject': 'URGENT: Verify your account immediately',
            'body': 'Your PayPal account has been suspended. Click here to verify: http://bit.ly/paypal-verify',
            'reply_to': 'noreply@suspicious-domain.com',
            'timestamp': datetime.now().isoformat()
        },
        {
            'sender': 'support@legitimate-company.com',
            'subject': 'Monthly Newsletter',
            'body': 'Thank you for subscribing to our newsletter. Here are this month\'s updates...',
            'reply_to': '',
            'timestamp': datetime.now().isoformat()
        },
        {
            'sender': 'winner@lottery-prize.net',
            'subject': 'Congratulations! You\'ve won $1,000,000',
            'body': 'You have been selected as our lucky winner! Claim your prize now by clicking here: http://192.168.1.100/claim',
            'reply_to': '',
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    print("=== Phishing-Aware Email Agent Analysis ===\n")
    
    for i, email_data in enumerate(test_emails, 1):
        print(f"--- Email {i} Analysis ---")
        result = agent.analyze_email(email_data)
        
        print(f"Email ID: {result.email_id}")
        print(f"Risk Score: {result.risk_score:.2f}")
        print(f"Suspicious: {'YES' if result.is_suspicious else 'NO'}")
        print(f"Patterns Detected: {len(result.detected_patterns)}")
        
        for pattern in result.detected_patterns:
            print(f"  - {pattern.pattern_type}: {pattern.pattern_value} (confidence: {pattern.confidence_score:.2f})")
        
        print("Recommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")
        
        print()
    
    # Show pattern statistics
    print("=== Pattern Learning Statistics ===")
    stats = agent.get_pattern_statistics()
    print(f"Total patterns learned: {stats['total_patterns']}")
    print(f"High confidence patterns: {stats['high_confidence_patterns']}")
    print(f"Pattern types: {stats['pattern_types']}")
    
    print("\nMost common patterns:")
    for pattern in stats['most_common_patterns']:
        print(f"  - {pattern['type']}: {pattern['value']} (seen {pattern['count']} times, confidence: {pattern['confidence']:.2f})")

if __name__ == '__main__':
    main()
