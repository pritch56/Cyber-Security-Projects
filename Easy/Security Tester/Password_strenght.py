from flask import Flask, render_template, request, jsonify
import json
import hashlib
import requests
import zxcvbn

app = Flask(__name__)

class PasswordStrengthEvaluator:
    def __init__(self, check_hibp=False):
        self.check_hibp = check_hibp

    def evaluate_password(self, password):
        zxcvbn_result = zxcvbn.zxcvbn(password)
        password_strength_score = zxcvbn_result['score']
        entropy = zxcvbn_result.get('entropy', None)
        crack_time = zxcvbn_result['crack_times_display']
        feedback = zxcvbn_result['feedback']
        patterns = self.extract_patterns(zxcvbn_result)

        breach_count = 0
        if self.check_hibp:
            breach_count = self.check_hibp_breach(password)

        return {
            "password_strength_score": password_strength_score,
            "entropy": entropy,
            "crack_time": crack_time,
            "feedback": feedback,
            "patterns": patterns,
            "breach_count": breach_count
        }

    def extract_patterns(self, zxcvbn_result):
        patterns = []
        for match in zxcvbn_result['sequence']:
            if 'pattern' in match:
                patterns.append(match['pattern'])
        return patterns

    def check_hibp_breach(self, password):
        sha1_hash = hashlib.sha1(password.encode('utf-8')).hexdigest().upper()
        prefix = sha1_hash[:5]
        suffix = sha1_hash[5:]

        url = f"https://api.pwnedpasswords.com/range/{prefix}"
        headers = {'User-Agent': 'FlaskPasswordStrengthApp/1.0'}
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                hashes = (line.split(':') for line in response.text.splitlines())
                for h, count in hashes:
                    if h == suffix:
                        return int(count)
        except requests.RequestException:
            # Timeout or other error, treat as 0 breach count
            pass
        return 0

evaluator = PasswordStrengthEvaluator(check_hibp=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        password = request.form.get("password", "")
        check_hibp = request.form.get("check_hibp") == "on"
        # Update evaluator setting based on form
        evaluator.check_hibp = check_hibp

        if not password:
            error = "Please enter a password to evaluate."
            return render_template("index.html", error=error, password="", check_hibp=check_hibp)

        result = evaluator.evaluate_password(password)

        # Prepare data for display
        return render_template(
            "index.html",
            password=password,
            result=result,
            check_hibp=check_hibp
        )

    # GET request
    return render_template("index.html", password="", result=None, check_hibp=True)

if __name__ == "__main__":
    app.run(debug=True)

