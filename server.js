const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.json());

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'pub', 'index.html'));
});

app.post('/analyze', (req, res) => {
    const contractCode = req.body.code;
    
    const pythonProcess = spawn('python3', ['./smardity/greedy_analyzer.py']);
    let result = '';
    
    pythonProcess.stdin.write(contractCode);
    pythonProcess.stdin.end();

    pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Analysis failed' });
        }
        try {
            const parsedResult = JSON.parse(result);
            res.json(parsedResult);
        } catch (e) {
            res.status(500).json({ error: 'Invalid analysis output' });
        }
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});