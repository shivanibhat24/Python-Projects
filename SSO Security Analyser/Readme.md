# SSO Security Testing Tool

A comprehensive automated security testing framework for Single Sign-On (SSO) implementations. This tool uses Selenium to identify common security vulnerabilities in authentication systems.

## Features

🔒 **Comprehensive Security Testing**: Tests for 8 critical SSO vulnerabilities  
🚀 **Fully Automated**: Runs end-to-end with minimal configuration  
📊 **Detailed Reporting**: Generates comprehensive JSON reports and logs  
⚙️ **Highly Configurable**: Customizable for different SSO implementations  
🔄 **CI/CD Integration**: Easy to integrate into your development pipeline  

## Security Tests

| Test | Description |
|------|-------------|
| CSRF Protection | Detects missing Cross-Site Request Forgery protections |
| Token Leakage | Identifies authentication tokens exposed in URLs |
| Open Redirect | Tests redirect_uri parameter validation |
| IDOR | Checks for Insecure Direct Object References across user accounts |
| OAuth State Validation | Verifies proper state parameter handling |
| JWT Validation | Tests for JWT algorithm vulnerabilities |
| Session Fixation | Ensures session IDs change after authentication |
| Brute Force Protection | Checks for rate limiting and account lockout mechanisms |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sso-security-testing.git
cd sso-security-testing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- Selenium
- Chrome WebDriver
- Chrome browser

## Usage

### Basic Usage

```bash
python sso_security_test.py --url https://your-sso-service.com
```

### With Custom Configuration

```bash
python sso_security_test.py --url https://your-sso-service.com --config config.json
```

## Configuration

Create a `config.json` file to customize the test parameters:

```json
{
  "headless": true,
  "login_page": "/login",
  "username_field": "username",
  "password_field": "password",
  "submit_button": "login-button",
  "success_indicator": "dashboard",
  "test_credentials": {
    "valid_user": {"username": "test_user", "password": "valid_password"},
    "admin_user": {"username": "admin", "password": "admin_password"}
  },
  "redirect_uri": "https://legitimate-app.com/callback",
  "client_id": "legitimate_client"
}
```

## Output and Reports

The tool generates:

1. **JSON Report**: `sso_security_report.json` with detailed findings
2. **Log File**: `sso_security_test.log` with test execution details
3. **Console Output**: Summary of vulnerabilities with severity ratings

Sample report:

```json
{
  "vulnerabilities_found": [
    {
      "name": "Token Leakage in URLs",
      "description": "Authentication tokens found in URL parameters",
      "evidence": "[{\"url\":\"https://example.com/dashboard?token=abc123\",\"parameter\":\"token\",\"value_preview\":\"abc123...\"}]"
    }
  ],
  "tests_performed": [
    {
      "name": "CSRF Protection",
      "timestamp": "2025-03-11 10:15:22",
      "result": "SECURE"
    },
    {
      "name": "Token Leakage in URLs",
      "timestamp": "2025-03-11 10:15:24",
      "result": "VULNERABLE"
    }
  ],
  "summary": {
    "total_tests": 8,
    "total_vulnerabilities": 1,
    "security_score": 87.5
  }
}
```

## Integration with CI/CD

Add this to your GitHub Actions workflow:

```yaml
name: SSO Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run SSO security tests
      run: |
        python sso_security_test.py --url ${{ secrets.SSO_TEST_URL }} --config config.json
    - name: Upload security report
      uses: actions/upload-artifact@v3
      with:
        name: security-report
        path: sso_security_report.json
```

## Customization

### Adding New Tests

Extend the `SSOSecurityTester` class with your own test methods:

```python
def test_your_custom_vulnerability(self):
    """Test for your custom vulnerability"""
    test_name = "Custom Vulnerability Test"
    try:
        # Your test logic here
        is_vulnerable = False  # Set based on your test results
        details = "Description of the vulnerability"
        evidence = "Evidence of the vulnerability"
        
        self._add_test_result(test_name, is_vulnerable, details, evidence)
    except Exception as e:
        logger.error(f"Error testing custom vulnerability: {e}")
        self._add_test_result(test_name, True, f"Error testing custom vulnerability: {e}")
```

### Configuration Options

| Option | Description |
|--------|-------------|
| `headless` | Run tests without visible browser window |
| `login_page` | Path to the login page |
| `username_field` | ID of the username input field |
| `password_field` | ID of the password input field |
| `submit_button` | ID of the login submit button |
| `success_indicator` | Class name indicating successful login |
| `test_credentials` | Valid credentials for testing |
| `redirect_uri` | Valid redirect URI for OAuth testing |
| `client_id` | Valid client ID for OAuth testing |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Disclaimer

This tool is intended for security testing of your own systems with proper authorization. Do not use it against systems you don't own or don't have permission to test.
