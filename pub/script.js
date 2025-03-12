async function analyzezeContract() {
    const code = document.getElementById('contractCode').value;
    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: code })
    });

    const results = await response.json();
    displayResults(results);
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    if (data.vulnerabilities.length === 0) {
        resultsDiv.innerHTML = '<p>No vulnerabilities found!</p>';
        return;
    }

    data.vulnerabilities.forEach(vuln => {
        const vulnDiv = document.createElement('div');
        vulnDiv.innerHTML = `
            <h3>${vuln.type}</h3>
            <p>Lines: ${vuln.lines.join(', ')}</p>
        `;
        resultsDiv.appendChild(vulnDiv);
    });
}