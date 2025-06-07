from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

HIBP_API_RANGE_URL = "https://api.pwnedpasswords.com/range/"

@app.route('/api/check_pwned', methods=['POST'])
def check_pwned():
    data = request.get_json()
    prefix = data.get('prefix')
    suffix = data.get('suffix').upper()

    if not prefix or not suffix or len(prefix) != 5 or len(suffix) != 35:
        return jsonify({'error': 'Invalid hash parts'}), 400

    try:
        response = requests.get(HIBP_API_RANGE_URL + prefix)
        if response.status_code != 200:
            return jsonify({'error': 'HIBP API request failed'}), 500

        hashes = response.text.splitlines()
        for line in hashes:
            line_suffix, count = line.split(':')
            if line_suffix == suffix:
                return jsonify({'count': int(count)})

        return jsonify({'count': 0})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)

