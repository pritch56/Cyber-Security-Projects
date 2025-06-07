import re
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify

app = Flask(__name__)

# Heuristic Analysis Function
def heuristic_analysis(url):
    score = 0
    red_flags = []

    # Check for IP address instead of domain name
    if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', url):
        score += 2
        red_flags.append("IP address instead of domain name")

    # Excessive use of special characters
    if len(re.findall(r'[@\-\/]', url)) > 3:
        score += 2
        red_flags.append("Excessive special characters")

    # Long subdomain chains
    subdomains = url.split('.')
    if len(subdomains) > 3:
        score += 2
        red_flags.append("Long subdomain chain")

    # Known deceptive keywords
    deceptive_keywords = ["login", "secure", "update"]
    if any(keyword in url for keyword in deceptive_keywords):
        score += 3
        red_flags.append("Contains deceptive keywords")

    return score, red_flags

# Blacklist Checking Function (Google Safe Browsing)
def check_blacklist(url, api_key):
    endpoint = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
    payload = {
        "client": {
            "clientId": "yourcompanyname",
            "clientVersion": "1.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    response = requests.post(f"{endpoint}?key={api_key}", json=payload)
    return response.json()

# HTML Content Inspection Function
def inspect_html(url):
    findings = []
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Check for suspicious <form> actions
        for form in soup.find_all('form'):
            action = form.get('action')
            if action and not action.startswith(url):
                findings.append(f"Suspicious form action: {action}")

        # Check for embedded JavaScript redirects
        scripts = soup.find_all('script')
        for script in scripts:
            if 'window.location' in script.text:
                findings.append("Embedded JavaScript redirect found")

        # Check for hidden iframes
        iframes = soup.find_all('iframe', style=re.compile('display:none'))
        for iframe in iframes:
            findings.append("Hidden iframe found")

    except requests.RequestException:
        findings.append("Could not access URL")

    return findings

# Main URL Safety Analysis Function
def analyze_url(url, api_key):
    heuristic_score, heuristic_flags = heuristic_analysis(url)
    blacklist_results = check_blacklist(url, api_key)
    html_findings = inspect_html(url)

    # Determine final risk level
    risk_level = "low"
    if heuristic_score > 5 or blacklist_results.get('matches'):
        risk_level = "high"
    elif heuristic_score > 2:
        risk_level = "medium"

    return {
        "url": url,
        "heuristic_score": heuristic_score,
        "heuristic_flags": heuristic_flags,
        "blacklist_hits": blacklist_results,
        "html_findings": html_findings,
        "final_risk_level": risk_level
    }

# Flask Route for URL Analysis
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    url = data.get('url')
    api_key = data.get('api_key')

    if not url or not api_key:
        return jsonify({"error": "URL and API key are required"}), 400

    result = analyze_url(url, api_key)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
