from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import logging
import argparse
import json
import re
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sso_security_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SSOSecurityTester:
    def __init__(self, base_url, config_file=None):
        self.base_url = base_url
        self.config = self._load_config(config_file)
        self.results = {
            "vulnerabilities_found": [],
            "tests_performed": [],
            "summary": {}
        }
        
        # Setup WebDriver
        chrome_options = Options()
        if self.config.get("headless", True):
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def _load_config(self, config_file):
        default_config = {
            "headless": True,
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
        
        if config_file:
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    return {**default_config, **user_config}
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        
        return default_config

    def run_all_tests(self):
        """Run all security tests and generate a report"""
        try:
            self.test_csrf_protection()
            self.test_token_leakage()
            self.test_open_redirect()
            self.test_idor()
            self.test_improper_state_validation()
            self.test_jwt_validation()
            self.test_session_fixation()
            self.test_brute_force_protection()
            
            # Generate summary
            total_tests = len(self.results["tests_performed"])
            total_vulnerabilities = len(self.results["vulnerabilities_found"])
            
            self.results["summary"] = {
                "total_tests": total_tests,
                "total_vulnerabilities": total_vulnerabilities,
                "security_score": ((total_tests - total_vulnerabilities) / total_tests) * 100 if total_tests > 0 else 0
            }
            
            # Output results
            self._generate_report()
            
        finally:
            self.driver.quit()
    
    def _generate_report(self):
        """Generate a detailed report of the test results"""
        report_file = "sso_security_report.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Report generated: {report_file}")
        logger.info(f"Security Score: {self.results['summary']['security_score']:.2f}%")
        logger.info(f"Vulnerabilities found: {len(self.results['vulnerabilities_found'])}")
        
        if self.results["vulnerabilities_found"]:
            logger.warning("VULNERABILITIES FOUND:")
            for vuln in self.results["vulnerabilities_found"]:
                logger.warning(f"- {vuln['name']}: {vuln['description']}")
    
    def _add_test_result(self, test_name, is_vulnerable, details="", evidence=""):
        """Add a test result to the results dictionary"""
        self.results["tests_performed"].append({
            "name": test_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": "VULNERABLE" if is_vulnerable else "SECURE"
        })
        
        if is_vulnerable:
            self.results["vulnerabilities_found"].append({
                "name": test_name,
                "description": details,
                "evidence": evidence
            })
            logger.warning(f"VULNERABLE: {test_name} - {details}")
        else:
            logger.info(f"SECURE: {test_name}")
    
    def login(self):
        """Perform login using test credentials"""
        try:
            login_url = self.base_url + self.config["login_page"]
            self.driver.get(login_url)
            
            # Enter credentials
            credentials = self.config["test_credentials"]["valid_user"]
            username_field = self.driver.find_element(By.ID, self.config["username_field"])
            password_field = self.driver.find_element(By.ID, self.config["password_field"])
            submit_button = self.driver.find_element(By.ID, self.config["submit_button"])
            
            username_field.send_keys(credentials["username"])
            password_field.send_keys(credentials["password"])
            submit_button.click()
            
            # Wait for successful login
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, self.config["success_indicator"])))
            
            # Get tokens or cookies for further tests
            tokens = {
                "cookies": self.driver.get_cookies(),
                "local_storage": self.driver.execute_script("return Object.keys(localStorage).reduce((obj, key) => { obj[key] = localStorage.getItem(key); return obj; }, {});"),
                "session_storage": self.driver.execute_script("return Object.keys(sessionStorage).reduce((obj, key) => { obj[key] = sessionStorage.getItem(key); return obj; }, {});")
            }
            
            return True, tokens
        
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False, None
    
    def test_csrf_protection(self):
        """Test if CSRF protections are in place"""
        test_name = "CSRF Protection"
        try:
            # Get the login page to check for CSRF tokens
            self.driver.get(self.base_url + self.config["login_page"])
            page_source = self.driver.page_source
            
            # Look for common CSRF token patterns
            csrf_patterns = [
                r'<input[^>]*name=["\'](csrf|_token|csrf_token|authenticity_token)["\'][^>]*>',
                r'<meta[^>]*name=["\'](csrf-token)["\'][^>]*>'
            ]
            
            has_csrf_token = any(re.search(pattern, page_source, re.IGNORECASE) for pattern in csrf_patterns)
            
            # Check request headers on form submission
            headers_script = """
            var observer = new PerformanceObserver((list) => {
                window.lastHeaders = list.getEntries()[0].toJSON();
            });
            observer.observe({entryTypes: ['resource']});
            """
            self.driver.execute_script(headers_script)
            
            # Submit the form
            try:
                credentials = self.config["test_credentials"]["valid_user"]
                username_field = self.driver.find_element(By.ID, self.config["username_field"])
                password_field = self.driver.find_element(By.ID, self.config["password_field"])
                submit_button = self.driver.find_element(By.ID, self.config["submit_button"])
                
                username_field.send_keys(credentials["username"])
                password_field.send_keys(credentials["password"])
                submit_button.click()
                
                time.sleep(2)
                
                # Get the headers that were sent
                headers = self.driver.execute_script("return window.lastHeaders;")
                has_csrf_header = headers and any("csrf" in str(key).lower() for key in headers)
            except:
                has_csrf_header = False
            
            is_vulnerable = not (has_csrf_token or has_csrf_header)
            details = "No CSRF protection mechanisms detected in login form"
            evidence = f"CSRF token in form: {has_csrf_token}, CSRF header: {has_csrf_header}"
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing CSRF protection: {e}")
            self._add_test_result(test_name, True, f"Error testing CSRF protection: {e}")
    
    def test_token_leakage(self):
        """Test for token leakage in URL parameters"""
        test_name = "Token Leakage in URLs"
        try:
            success, tokens = self.login()
            if not success:
                self._add_test_result(test_name, True, "Could not test token leakage - login failed")
                return
            
            # Check all URLs after login for token parameters
            token_patterns = ["token", "access_token", "id_token", "auth", "jwt"]
            vulnerable_urls = []
            
            # Navigate to a few pages to capture URLs
            pages_to_check = ["", "/profile", "/dashboard", "/settings"]
            for page in pages_to_check:
                try:
                    self.driver.get(self.base_url + page)
                    current_url = self.driver.current_url
                    
                    # Check for tokens in URL
                    parsed_url = urllib.parse.urlparse(current_url)
                    query_params = urllib.parse.parse_qs(parsed_url.query)
                    
                    for param, value in query_params.items():
                        if any(token_str in param.lower() for token_str in token_patterns):
                            vulnerable_urls.append({
                                "url": current_url,
                                "parameter": param,
                                "value_preview": value[0][:10] + "..." if value and value[0] else ""
                            })
                except:
                    continue
            
            is_vulnerable = len(vulnerable_urls) > 0
            details = "Authentication tokens found in URL parameters (risk of token leakage in browser history, referrers, logs)"
            evidence = json.dumps(vulnerable_urls)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing token leakage: {e}")
            self._add_test_result(test_name, True, f"Error testing token leakage: {e}")
    
    def test_open_redirect(self):
        """Test for open redirect vulnerabilities"""
        test_name = "Open Redirect"
        try:
            # Test various redirect_uri manipulations
            malicious_redirects = [
                "https://attacker.com/callback",
                "https://attacker.com@legitimate-app.com",
                f"{self.base_url}@attacker.com", 
                f"https://attacker.com?{self.config['redirect_uri']}",
                "javascript:alert('XSS')",
                "data:text/html,<script>alert('XSS')</script>"
            ]
            
            successful_redirects = []
            
            for redirect in malicious_redirects:
                try:
                    auth_url = f"{self.base_url}/auth?client_id={self.config['client_id']}&redirect_uri={urllib.parse.quote(redirect)}"
                    self.driver.get(auth_url)
                    
                    # Wait a moment for any redirects to complete
                    time.sleep(2)
                    current_url = self.driver.current_url
                    
                    # Check if we were redirected to the malicious URI or part of it
                    if any(part in current_url for part in ["attacker.com", "javascript:", "data:"]):
                        successful_redirects.append({
                            "payload": redirect,
                            "result_url": current_url
                        })
                except:
                    continue
            
            is_vulnerable = len(successful_redirects) > 0
            details = "Open redirect vulnerability found - malicious redirect_uri parameters were accepted"
            evidence = json.dumps(successful_redirects)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing open redirect: {e}")
            self._add_test_result(test_name, True, f"Error testing open redirect: {e}")
    
    def test_idor(self):
        """Test for Insecure Direct Object References"""
        test_name = "Insecure Direct Object References (IDOR)"
        try:
            success, _ = self.login()
            if not success:
                self._add_test_result(test_name, True, "Could not test IDOR - login failed")
                return
            
            # Look for potential user IDs in the page
            user_id_patterns = [
                r'user[_\-]id["\']?\s*[=:]\s*["\']?(\d+)["\']?',
                r'user["\']?\s*[=:]\s*["\']?(\d+)["\']?',
                r'id["\']?\s*[=:]\s*["\']?(\d+)["\']?'
            ]
            
            # Find some potential user IDs
            page_source = self.driver.page_source
            found_ids = []
            
            for pattern in user_id_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    found_ids.extend(matches)
            
            # If no IDs found, try some common ones
            if not found_ids:
                found_ids = ["1", "2", "10", "100"]
            
            # Try to access other user profiles
            vulnerable_endpoints = []
            current_user_content = None
            
            # First get the current user's profile content as baseline
            try:
                profile_url = f"{self.base_url}/profile"
                self.driver.get(profile_url)
                current_user_content = self.driver.page_source
            except:
                pass
            
            # Now try other user IDs
            for user_id in found_ids:
                try:
                    # Test different URL patterns
                    endpoints = [
                        f"/profile/{user_id}",
                        f"/user/{user_id}",
                        f"/users/{user_id}",
                        f"/profile?id={user_id}",
                        f"/api/user/{user_id}"
                    ]
                    
                    for endpoint in endpoints:
                        self.driver.get(f"{self.base_url}{endpoint}")
                        time.sleep(1)
                        
                        # If we get a 200 OK and different content than current user
                        if "404" not in self.driver.title and "error" not in self.driver.title.lower():
                            if current_user_content and self.driver.page_source != current_user_content:
                                vulnerable_endpoints.append({
                                    "user_id": user_id,
                                    "endpoint": endpoint,
                                    "status": "200 OK"
                                })
                except:
                    continue
            
            is_vulnerable = len(vulnerable_endpoints) > 0
            details = "IDOR vulnerability found - could access other user profiles by changing IDs"
            evidence = json.dumps(vulnerable_endpoints)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing IDOR: {e}")
            self._add_test_result(test_name, True, f"Error testing IDOR: {e}")
    
    def test_improper_state_validation(self):
        """Test for improper state parameter validation in OAuth flow"""
        test_name = "OAuth State Parameter Validation"
        try:
            # Attempt to complete OAuth flow with various state manipulations
            auth_url = f"{self.base_url}/auth?client_id={self.config['client_id']}&redirect_uri={urllib.parse.quote(self.config['redirect_uri'])}"
            
            state_tests = [
                {"name": "No state parameter", "url": f"{auth_url}"},
                {"name": "Empty state parameter", "url": f"{auth_url}&state="},
                {"name": "Invalid state parameter", "url": f"{auth_url}&state=invalid_state_value"}
            ]
            
            successful_auths = []
            
            for test in state_tests:
                try:
                    self.driver.get(test["url"])
                    
                    # Try to complete the flow with login
                    try:
                        username_field = self.driver.find_element(By.ID, self.config["username_field"])
                        password_field = self.driver.find_element(By.ID, self.config["password_field"])
                        submit_button = self.driver.find_element(By.ID, self.config["submit_button"])
                        
                        credentials = self.config["test_credentials"]["valid_user"]
                        username_field.send_keys(credentials["username"])
                        password_field.send_keys(credentials["password"])
                        submit_button.click()
                        
                        # Wait to see if we get redirected successfully
                        time.sleep(3)
                        
                        # If we reach the redirect_uri with a code, it's vulnerable
                        current_url = self.driver.current_url
                        if self.config["redirect_uri"] in current_url and "code=" in current_url:
                            successful_auths.append({
                                "test": test["name"],
                                "result_url": current_url
                            })
                    except:
                        continue
                except:
                    continue
            
            is_vulnerable = len(successful_auths) > 0
            details = "OAuth state parameter validation issue - authentication flow completed with manipulated state"
            evidence = json.dumps(successful_auths)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing state validation: {e}")
            self._add_test_result(test_name, True, f"Error testing state validation: {e}")
    
    def test_jwt_validation(self):
        """Test for JWT validation issues"""
        test_name = "JWT Token Validation"
        try:
            success, tokens = self.login()
            if not success:
                self._add_test_result(test_name, True, "Could not test JWT validation - login failed")
                return
            
            # Extract potential JWT tokens from cookies, local storage, session storage
            jwt_tokens = []
            
            # Check cookies
            for cookie in tokens["cookies"]:
                if self._is_jwt(cookie["value"]):
                    jwt_tokens.append({"source": "cookie", "name": cookie["name"], "value": cookie["value"]})
            
            # Check local and session storage
            for storage_type in ["local_storage", "session_storage"]:
                for key, value in tokens[storage_type].items():
                    if isinstance(value, str) and self._is_jwt(value):
                        jwt_tokens.append({"source": storage_type, "name": key, "value": value})
            
            if not jwt_tokens:
                self._add_test_result(test_name, False, "No JWT tokens found to test")
                return
            
            # Test the "none" algorithm vulnerability
            vulnerable_tokens = []
            
            for jwt_info in jwt_tokens:
                # Create a none algorithm token - basic implementation
                # For a full test, a proper JWT library should be used
                try:
                    token_parts = jwt_info["value"].split('.')
                    if len(token_parts) != 3:
                        continue
                    
                    # Decode header and modify algorithm
                    import base64
                    header = json.loads(self._decode_base64(token_parts[0]))
                    payload = json.loads(self._decode_base64(token_parts[1]))
                    
                    # Create a none algorithm token
                    header["alg"] = "none"
                    new_header = self._encode_base64(json.dumps(header))
                    new_payload = self._encode_base64(json.dumps(payload))
                    
                    # Create the token without signature
                    none_token = f"{new_header}.{new_payload}."
                    
                    # Try to use the token
                    if jwt_info["source"] == "cookie":
                        self.driver.add_cookie({"name": jwt_info["name"], "value": none_token})
                    else:
                        self.driver.execute_script(f"{jwt_info['source'].split('_')[0]}Storage.setItem('{jwt_info['name']}', '{none_token}')")
                    
                    # Try to access a protected page
                    self.driver.get(f"{self.base_url}/dashboard")
                    time.sleep(2)
                    
                    # Check if we're still authenticated
                    try:
                        self.driver.find_element(By.CLASS_NAME, self.config["success_indicator"])
                        vulnerable_tokens.append({
                            "source": jwt_info["source"],
                            "name": jwt_info["name"],
                            "vulnerability": "none algorithm"
                        })
                    except:
                        pass
                except:
                    continue
            
            is_vulnerable = len(vulnerable_tokens) > 0
            details = "JWT validation issues found - tokens with 'none' algorithm were accepted"
            evidence = json.dumps(vulnerable_tokens)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing JWT validation: {e}")
            self._add_test_result(test_name, True, f"Error testing JWT validation: {e}")
    
    def _is_jwt(self, token_string):
        """Check if a string is likely a JWT token"""
        if not isinstance(token_string, str):
            return False
        
        # JWT pattern: xxxxx.yyyyy.zzzzz
        pattern = r'^[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+$'
        return bool(re.match(pattern, token_string))
    
    def _decode_base64(self, data):
        """Decode base64 with padding handling"""
        data += '=' * (4 - len(data) % 4) if len(data) % 4 != 0 else ''
        return base64.b64decode(data).decode('utf-8')
    
    def _encode_base64(self, data):
        """Encode to base64 and remove padding"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        return base64.b64encode(data).decode('utf-8').rstrip('=')
    
    def test_session_fixation(self):
        """Test for session fixation vulnerabilities"""
        test_name = "Session Fixation"
        try:
            # Step 1: Get a session identifier before authentication
            self.driver.get(self.base_url)
            
            # Get session cookies before login
            pre_auth_cookies = {cookie["name"]: cookie["value"] for cookie in self.driver.get_cookies()}
            
            # Step 2: Authenticate
            success, _ = self.login()
            if not success:
                self._add_test_result(test_name, True, "Could not test session fixation - login failed")
                return
            
            # Step 3: Get session cookies after authentication
            post_auth_cookies = {cookie["name"]: cookie["value"] for cookie in self.driver.get_cookies()}
            
            # Step 4: Check if session identifiers changed after authentication
            unchanged_session_cookies = {}
            for name, value in pre_auth_cookies.items():
                if name in post_auth_cookies and post_auth_cookies[name] == value:
                    # Session identifier didn't change after authentication
                    if any(session_key in name.lower() for session_key in ["session", "sid", "auth"]):
                        unchanged_session_cookies[name] = value
            
            is_vulnerable = len(unchanged_session_cookies) > 0
            details = "Session fixation vulnerability detected - session identifiers not changed after authentication"
            evidence = json.dumps(unchanged_session_cookies)
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing session fixation: {e}")
            self._add_test_result(test_name, True, f"Error testing session fixation: {e}")
    
    def test_brute_force_protection(self):
        """Test for brute force protection mechanisms"""
        test_name = "Brute Force Protection"
        try:
            # Test multiple incorrect login attempts
            self.driver.get(self.base_url + self.config["login_page"])
            
            # Use a valid username with invalid passwords
            username = self.config["test_credentials"]["valid_user"]["username"]
            
            # Try multiple invalid login attempts
            login_attempts = 5
            captcha_detected = False
            rate_limit_detected = False
            account_lockout_detected = False
            
            for i in range(login_attempts):
                try:
                    # Clear any existing input
                    try:
                        username_field = self.driver.find_element(By.ID, self.config["username_field"])
                        password_field = self.driver.find_element(By.ID, self.config["password_field"])
                        
                        username_field.clear()
                        password_field.clear()
                    except:
                        # If we can't clear the fields, refresh and try again
                        self.driver.refresh()
                        username_field = self.driver.find_element(By.ID, self.config["username_field"])
                        password_field = self.driver.find_element(By.ID, self.config["password_field"])
                    
                    # Enter credentials
                    username_field.send_keys(username)
                    password_field.send_keys(f"wrong_password_{i}")
                    
                    # Submit form
                    submit_button = self.driver.find_element(By.ID, self.config["submit_button"])
                    submit_button.click()
                    
                    # Wait a moment
                    time.sleep(1)
                    
                    # Check for CAPTCHA
                    page_source = self.driver.page_source.lower()
                    if any(captcha_term in page_source for captcha_term in ["captcha", "recaptcha", "hcaptcha", "prove you're human"]):
                        captcha_detected = True
                        break
                    
                    # Check for rate limiting
                    if any(rate_term in page_source for rate_term in ["too many", "rate limit", "try again later", "temporary block"]):
                        rate_limit_detected = True
                        break
                    
                    # Check for account lockout
                    if any(lockout_term in page_source for lockout_term in ["account locked", "account disabled", "locked out"]):
                        account_lockout_detected = True
                        break
                    
                except NoSuchElementException:
                    # If we can't find the login form, we might have been redirected or blocked
                    page_source = self.driver.page_source.lower()
                    if any(block_term in page_source for block_term in ["captcha", "too many", "rate limit", "account locked"]):
                        rate_limit_detected = True
                        break
                except:
                    continue
            
            # Try the valid login after brute force attempts
            try:
                self.driver.get(self.base_url + self.config["login_page"])
                
                username_field = self.driver.find_element(By.ID, self.config["username_field"])
                password_field = self.driver.find_element(By.ID, self.config["password_field"])
                
                credentials = self.config["test_credentials"]["valid_user"]
                username_field.send_keys(credentials["username"])
                password_field.send_keys(credentials["password"])
                
                submit_button = self.driver.find_element(By.ID, self.config["submit_button"])
                submit_button.click()
                
                # Wait a moment
                time.sleep(3)
                
                # Check if login was successful
                login_blocked = False
                try:
                    self.driver.find_element(By.CLASS_NAME, self.config["success_indicator"])
                except:
                    # If we can't find the success indicator, login might be blocked
                    login_blocked = True
            except:
                login_blocked = True
            
            protection_measures = []
            if captcha_detected:
                protection_measures.append("CAPTCHA")
            if rate_limit_detected:
                protection_measures.append("Rate limiting")
            if account_lockout_detected or login_blocked:
                protection_measures.append("Account lockout")
            
            is_vulnerable = len(protection_measures) == 0
            details = "No brute force protection mechanisms detected"
            if not is_vulnerable:
                details = f"Brute force protection detected: {', '.join(protection_measures)}"
            
            evidence = f"Login attempts: {login_attempts}, Protections: {protection_measures}"
            
            self._add_test_result(test_name, is_vulnerable, details, evidence)
            
        except Exception as e:
            logger.error(f"Error testing brute force protection: {e}")
            self._add_test_result(test_name, True, f"Error testing brute force protection: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSO Security Tester")
    parser.add_argument("--url", required=True, help="Base URL of the SSO service to test")
    parser.add_argument("--config", help="Path to configuration JSON file")
    
    args = parser.parse_args()
    
    tester = SSOSecurityTester(args.url, args.config)
    tester.run_all_tests()
