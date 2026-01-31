// app.js - Simplified and fixed version
document.addEventListener('DOMContentLoaded', function() {
    console.log('app.js loaded');
    
    // Global variables
    let mergedData = null;
    let trainData = null;
    let testData = null;
    let currentChart = null;
    
    // DOM Elements
    const elements = {
        loadBtn: document.getElementById('loadBtn'),
        resetBtn: document.getElementById('resetBtn'),
        runAnalysisBtn: document.getElementById('runAnalysisBtn'),
        exportCSVBtn: document.getElementById('exportCSVBtn'),
        exportJSONBtn: document.getElementById('exportJSONBtn'),
        exportReportBtn: document.getElementById('exportReportBtn'),
        trainFile: document.getElementById('trainFile'),
        testFile: document.getElementById('testFile'),
        dropZone: document.getElementById('dropZone'),
        previewBody: document.getElementById('previewBody'),
        visualizationChart: document.getElementById('visualizationChart'),
        missingValuesContainer: document.getElementById('missingValuesContainer'),
        statsSummary: document.getElementById('statsSummary'),
        exportStatus: document.getElementById('exportStatus'),
        totalPassengers: document.getElementById('totalPassengers'),
        trainCount: document.getElementById('trainCount'),
        testCount: document.getElementById('testCount'),
        featureCount: document.getElementById('featureCount')
    };
    
    // Configuration - SWAP THESE FOR OTHER DATASETS
    const datasetConfig = {
        features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        target: 'Survived',
        identifier: 'PassengerId',
        categorical: ['Sex', 'Embarked', 'Pclass'],
        numerical: ['Age', 'Fare', 'SibSp', 'Parch']
    };
    
    // Initialize
    init();
    
    function init() {
        console.log('Initializing dashboard...');
        
        // Setup event listeners
        setupEventListeners();
        
        // Setup drop zone
        setupDropZone();
        
        console.log('Dashboard initialized successfully');
    }
    
    function setupEventListeners() {
        console.log('Setting up event listeners...');
        
        // Load and merge data
        elements.loadBtn.addEventListener('click', loadAndMergeData);
        
        // Reset dashboard
        elements.resetBtn.addEventListener('click', resetDashboard);
        
        // Run analysis
        elements.runAnalysisBtn.addEventListener('click', runEDA);
        
        // Export buttons
        elements.exportCSVBtn.addEventListener('click', exportCSV);
        elements.exportJSONBtn.addEventListener('click', exportJSON);
        elements.exportReportBtn.addEventListener('click', generateReport);
        
        // Visualization buttons
        document.querySelectorAll('.viz-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const vizType = parseInt(this.getAttribute('data-viz'));
                generateVisualization(vizType);
            });
        });
        
        console.log('Event listeners set up');
    }
    
    function setupDropZone() {
        elements.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            elements.dropZone.style.borderColor = '#3498db';
            elements.dropZone.style.background = '#e8f4fc';
        });
        
        elements.dropZone.addEventListener('dragleave', () => {
            elements.dropZone.style.borderColor = '#bdc3c7';
            elements.dropZone.style.background = '#f8f9fa';
        });
        
        elements.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            elements.dropZone.style.borderColor = '#bdc3c7';
            elements.dropZone.style.background = '#f8f9fa';
            
            const files = e.dataTransfer.files;
            handleDroppedFiles(files);
        });
        
        elements.dropZone.addEventListener('click', () => {
            elements.trainFile.click();
        });
    }
    
    function handleDroppedFiles(files) {
        Array.from(files).forEach(file => {
            if (file.name.toLowerCase().includes('train')) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                elements.trainFile.files = dataTransfer.files;
                showStatus('Train file added: ' + file.name, 'success');
            } else if (file.name.toLowerCase().includes('test')) {
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                elements.testFile.files = dataTransfer.files;
                showStatus('Test file added: ' + file.name, 'success');
            }
        });
    }
    
    async function loadAndMergeData() {
        const trainFile = elements.trainFile.files[0];
        const testFile = elements.testFile.files[0];
        
        if (!trainFile || !testFile) {
            showAlert('Please upload both train.csv and test.csv files');
            return;
        }
        
        try {
            showStatus('Loading and merging datasets...', 'info');
            
            // Parse CSV files
            const trainPromise = parseCSV(trainFile);
            const testPromise = parseCSV(testFile);
            
            const [trainResults, testResults] = await Promise.all([trainPromise, testPromise]);
            
            // Add source column and store data
            trainData = trainResults.data.map(row => ({...row, source: 'train'}));
            testData = testResults.data.map(row => ({...row, source: 'test'}));
            
            // Merge datasets
            mergedData = [...trainData, ...testData];
            
            // Update UI
            updateOverview();
            showDataPreview();
            analyzeMissingValues();
            
            showStatus('Data loaded successfully! Total records: ' + mergedData.length, 'success');
            
        } catch (error) {
            console.error('Error:', error);
            showAlert('Error loading data: ' + error.message);
        }
    }
    
    function parseCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: resolve,
                error: reject
            });
        });
    }
    
    function updateOverview() {
        elements.totalPassengers.textContent = mergedData.length;
        elements.trainCount.textContent = trainData.length;
        elements.testCount.textContent = testData.length;
        elements.featureCount.textContent = datasetConfig.features.length;
    }
    
    function showDataPreview() {
        const previewBody = elements.previewBody;
        previewBody.innerHTML = '';
        
        // Show first 5 rows
        mergedData.slice(0, 5).forEach(row => {
            const tr = document.createElement('tr');
            
            const cells = [
                row.PassengerId || '',
                row.Name ? (row.Name.length > 20 ? row.Name.substring(0, 20) + '...' : row.Name) : '',
                row.Pclass || '',
                row.Sex || '',
                row.Age ? row.Age.toFixed(1) : 'N/A',
                row.Survived !== undefined ? row.Survived : 'N/A'
            ];
            
            cells.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                tr.appendChild(td);
            });
            
            previewBody.appendChild(tr);
        });
    }
    
    function analyzeMissingValues() {
        const container = elements.missingValuesContainer;
        container.innerHTML = '';
        
        datasetConfig.features.forEach(feature => {
            const missingCount = mergedData.filter(row => 
                row[feature] === null || row[feature] === undefined || row[feature] === ''
            ).length;
            
            const missingPercentage = (missingCount / mergedData.length * 100).toFixed(1);
            
            const barDiv = document.createElement('div');
            barDiv.className = 'missing-bar';
            
            barDiv.innerHTML = `
                <span class="missing-label">${feature}</span>
                <div class="missing-progress">
                    <div class="missing-fill" style="width: ${missingPercentage}%"></div>
                </div>
                <span style="min-width: 50px; text-align: right;">${missingPercentage}%</span>
            `;
            
            container.appendChild(barDiv);
        });
    }
    
    function runEDA() {
        if (!mergedData) {
            showAlert('Please load data first');
            return;
        }
        
        showStatus('Running Exploratory Data Analysis...', 'info');
        
        elements.statsSummary.innerHTML = '';
        
        // Create tabs
        const tabs = document.createElement('div');
        tabs.style.display = 'flex';
        tabs.style.gap = '10px';
        tabs.style.marginBottom = '20px';
        tabs.style.flexWrap = 'wrap';
        
        ['Numerical Stats', 'Categorical Stats', 'Survival Analysis'].forEach((name, idx) => {
            const tab = document.createElement('button');
            tab.textContent = name;
            tab.className = 'btn btn-secondary';
            tab.style.padding = '10px 20px';
            tab.onclick = () => showStatsTab(idx);
            tabs.appendChild(tab);
        });
        
        elements.statsSummary.appendChild(tabs);
        
        // Content container
        const content = document.createElement('div');
        content.id = 'statsContent';
        elements.statsSummary.appendChild(content);
        
        // Show first tab
        showStatsTab(0);
        
        showStatus('EDA completed!', 'success');
    }
    
    function showStatsTab(tabIndex) {
        const content = document.getElementById('statsContent');
        if (!content) return;
        
        content.innerHTML = '';
        
        switch(tabIndex) {
            case 0:
                showNumericalStats(content);
                break;
            case 1:
                showCategoricalStats(content);
                break;
            case 2:
                showSurvivalStats(content);
                break;
        }
    }
    
    function showNumericalStats(container) {
        const table = document.createElement('table');
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        
        table.innerHTML = `
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 12px; border: 1px solid #ddd;">Feature</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Mean</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Median</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Std Dev</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Min</th>
                    <th style="padding: 12px; border: 1px solid #ddd;">Max</th>
                </tr>
            </thead>
            <tbody>
        `;
        
        datasetConfig.numerical.forEach(feature => {
            const values = trainData
                .filter(row => row[feature] !== null && !isNaN(row[feature]))
                .map(row => parseFloat(row[feature]));
            
            if (values.length === 0) return;
            
            const mean = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2);
            const sorted = [...values].sort((a, b) => a - b);
            const median = sorted[Math.floor(sorted.length / 2)].toFixed(2);
            const std = calculateStdDev(values).toFixed(2);
            const min = Math.min(...values).toFixed(2);
            const max = Math.max(...values).toFixed(2);
            
            table.innerHTML += `
                <tr>
                    <td style="padding: 10px; border: 1px solid #ddd;"><strong>${feature}</strong></td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${mean}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${median}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${std}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${min}</td>
                    <td style="padding: 10px; border: 1px solid #ddd;">${max}</td>
                </tr>
            `;
        });
        
        table.innerHTML += '</tbody>';
        container.appendChild(table);
    }
    
    function calculateStdDev(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
        const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
        return Math.sqrt(variance);
    }
    
    function showCategoricalStats(container) {
        datasetConfig.categorical.forEach(feature => {
            const card = document.createElement('div');
            card.style.background = '#f8f9fa';
            card.style.padding = '15px';
            card.style.borderRadius = '8px';
            card.style.marginBottom = '15px';
            
            const counts = {};
            trainData.forEach(row => {
                const value = row[feature] || 'Missing';
                counts[value] = (counts[value] || 0) + 1;
            });
            
            let content = `<h4 style="margin-bottom: 10px; color: #2c3e50;">${feature} Distribution</h4>`;
            
            Object.entries(counts).forEach(([value, count]) => {
                const percentage = ((count / trainData.length) * 100).toFixed(1);
                content += `
                    <div style="display: flex; justify-content: space-between; margin-bottom: 5px; padding: 5px 0; border-bottom: 1px solid #ddd;">
                        <span>${value}:</span>
                        <span><strong>${count}</strong> (${percentage}%)</span>
                    </div>
                `;
            });
            
            card.innerHTML = content;
            container.appendChild(card);
        });
    }
    
    function showSurvivalStats(container) {
        const survived = trainData.filter(row => row.Survived === 1).length;
        const survivalRate = ((survived / trainData.length) * 100).toFixed(1);
        
        let html = `
            <div style="text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
                <div style="font-size: 2.5rem; font-weight: bold;">${survivalRate}%</div>
                <div style="font-size: 1.2rem;">Overall Survival Rate</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">${survived} out of ${trainData.length} passengers survived</div>
            </div>
        `;
        
        // Survival by Sex
        const genderData = { male: { total: 0, survived: 0 }, female: { total: 0, survived: 0 } };
        trainData.forEach(row => {
            if (row.Sex === 'male') {
                genderData.male.total++;
                if (row.Survived === 1) genderData.male.survived++;
            } else if (row.Sex === 'female') {
                genderData.female.total++;
                if (row.Survived === 1) genderData.female.survived++;
            }
        });
        
        html += '<h3 style="color: #2c3e50; margin-bottom: 15px;">Survival by Gender</h3>';
        html += '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 25px;">';
        
        ['male', 'female'].forEach(gender => {
            const rate = ((genderData[gender].survived / genderData[gender].total) * 100).toFixed(1);
            html += `
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                    <div style="font-weight: bold; margin-bottom: 5px;">${gender.charAt(0).toUpperCase() + gender.slice(1)}</div>
                    <div style="font-size: 1.5rem; color: #27ae60;">${rate}%</div>
                    <div style="font-size: 0.9rem; color: #666;">${genderData[gender].survived}/${genderData[gender].total}</div>
                </div>
            `;
        });
        
        html += '</div>';
        
        container.innerHTML = html;
    }
    
    function generateVisualization(type) {
        if (!mergedData) {
            showAlert('Please load data first');
            return;
        }
        
        if (currentChart) {
            currentChart.destroy();
        }
        
        const ctx = elements.visualizationChart.getContext('2d');
        
        switch(type) {
            case 1:
                createGenderChart(ctx);
                break;
            case 2:
                createClassChart(ctx);
                break;
            case 3:
                createEmbarkationChart(ctx);
                break;
            case 4:
                createAgeHistogram(ctx);
                break;
            case 5:
                createFareHistogram(ctx);
                break;
            case 6:
                createCorrelationHeatmap(ctx);
                break;
        }
        
        showStatus(`Visualization ${type} generated`, 'success');
    }
    
    function createGenderChart(ctx) {
        const genderData = { male: { survived: 0, total: 0 }, female: { survived: 0, total