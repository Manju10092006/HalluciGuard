import urllib.request
import json
import os
api_key = os.environ.get('API_KEY', 'YOUR_API_KEY')
url = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        print('Available models:')
        for m in data.get('models', []):
            if 'flash' in m.get('name', '').lower() and 'generateContent' in m.get('supportedGenerationMethods', []):
                print(f"{m.get('name')} - {m.get('displayName')}")
except Exception as e:
    print(f'Error: {e}')
