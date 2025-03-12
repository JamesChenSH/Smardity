import sys
import json

def analyze_contract(code):
    # Replace this with your actual vulnerability analysis logic
    # This is a mock implementation
    vulnerabilities = []
    
    # Example detection logic
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'unsafe' in line:
            vulnerabilities.append({
                'type': 'Mock Vulnerability',
                'lines': [i+1]  # Lines are 1-indexed
            })
    
    return {
        'vulnerabilities': vulnerabilities
    }

if __name__ == '__main__':
    code = sys.stdin.read()
    result = analyze_contract(code)
    print(json.dumps(result))