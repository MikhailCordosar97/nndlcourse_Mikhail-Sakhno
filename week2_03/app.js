// app.js - Fixed version with correct element IDs
"use strict";

// ==================== GLOBAL VARIABLES ====================
let rawData = { train: null, test: null };
let processedData = {
    X_train: null,
    y_train: null,
    X_test: null,
    featureNames: [],
    scalers: {}
};
let model = null;
let trainingHistory = [];
let predictions = null;
let validationData = null;
let currentThreshold = 0.5;
let isTraining = false;
let trainingChart = null;
let rocChart = null;
let confusionMatrixChart = null;

// ==================== UTILITY FUNCTIONS ====================
function showStatus(message, type = 'info', elementId = 'data-status') {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element #${elementId} not found!`);
        return;
    }
    
    element.textContent = message;
    element.className = 'status';
    
    switch(type) {
        case 'success':
            element.classList.add('status-success');
            break;
        case 'error':
            element.classList.add('status-error');
            break;
        case 'info':
            element.classList.add('status-info');
            break;
    }
    
    console.log(`${type.toUpperCase()}: ${message}`);
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                console.log(`Parsed ${results.data.length} rows`);
                resolve(results.data);
            },
            error: (error) => reject(error)
        });
    });
}

// ==================== 1. DATA LOADING & EXPLORATION ====================
async function loadData() {
    console.log("loadData function called!");
    
    const trainFileInput = document.getElementById('train-file');
    const testFileInput = document.getElementById('test-file');
    
    console.log("Train file:", trainFileInput?.files[0]);
    console.log("Test file:", testFileInput?.files[0]);
    
    if (!trainFileInput || !trainFileInput.files[0] || !testFileInput || !testFileInput.files[0]) {
        showStatus('❌ Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    showStatus('⏳ Loading CSV files...', 'info');
    
    try {
        // Read files
        const trainText = await readFile(trainFileInput.files[0]);
        const testText = await readFile(testFileInput.files[0]);
        
        console.log("Train text length:", trainText.length);
        console.log("Test text length:", testText.length);
        
        // Parse CSV
        const trainResults = await parseCSV(trainText);
        const testResults = await parseCSV(testText);
        
        rawData.train = trainResults;
        rawData.test = testResults;
        
        console.log("Train data loaded:", rawData.train.length, "rows");
        console.log("Test data loaded:", rawData.test.length, "rows");
        console.log("First train row:", rawData.train[0]);
        
        showStatus(`✅ Loaded ${rawData.train.length} training and ${rawData.test.length} test samples`, 'success');
        
        // Update UI
        updateDataOverview();
        createInitialChart();
        
        // Enable preprocessing controls
        const preprocessControls = document.getElementById('preprocess-controls');
        if (preprocessControls) {
            preprocessControls.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

function updateDataOverview() {
    console.log("Updating data overview...");
    
    // Update statistics
    const totalPassengers = document.getElementById('total-passengers');
    const trainCount = document.getElementById('train-count');
    const testCount = document.getElementById('test-count');
    const survivalRate = document.getElementById('survival-rate');
    
    if (totalPassengers) {
        totalPassengers.textContent = rawData.train.length + rawData.test.length;
    }
    if (trainCount) {
        trainCount.textContent = rawData.train.length;
    }
    if (testCount) {
        testCount.textContent = rawData.test.length;
    }
    
    // Calculate survival rate
    const survived = rawData.train.filter(row => row.Survived === 1).length;
    const survivalRateValue = ((survived / rawData.train.length) * 100).toFixed(1);
    if (survivalRate) {
        survivalRate.textContent = `${survivalRateValue}%`;
    }
    
    // Show data preview
    showDataPreview();
    
    // Analyze missing values
    showMissingValues();
    
    // Show the overview section
    const dataOverview = document.getElementById('data-overview');
    if (dataOverview) {
        dataOverview.style.display = 'block';
    }
}

function showDataPreview() {
    const tbody = document.getElementById('preview-body');
    if (!tbody) {
        console.error("Element #preview-body not found!");
        return;
    }
    
    tbody.innerHTML = '';
    
    // Show first 5 rows from training data
    const previewData = rawData.train.slice(0, 5);
    
    previewData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.PassengerId || ''}</td>
            <td>${(row.Name || '').substring(0, 20)}${row.Name && row.Name.length > 20 ? '...' : ''}</td>
            <td>${row.Pclass || ''}</td>
            <td>${row.Sex || ''}</td>
            <td>${row.Age ? row.Age.toFixed(1) : 'N/A'}</td>
            <td>${row.Survived !== undefined ? row.Survived : 'N/A'}</td>
        `;
        tbody.appendChild(tr);
    });
}

function showMissingValues() {
    const container = document.getElementById('missing-values');
    if (!container) {
        console.error("Element #missing-values not found!");
        return;
    }
    
    const features = ['Age', 'Cabin', 'Embarked', 'Fare'];
    
    let html = '';
    features.forEach(feature => {
        const missing = rawData.train.filter(row => 
            row[feature] === null || row[feature] === undefined || row[feature] === ''
        ).length;
        
        const percentage = ((missing / rawData.train.length) * 100).toFixed(1);
        
        html += `
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: 500;">${feature}</span>
                    <span>${percentage}% (${missing})</span>
                </div>
                <div style="height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                    <div style="height: 100%; width: ${percentage}%; background: linear-gradient(90deg, #ef4444, #f59e0b);"></div>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function createInitialChart() {
    const ctx = document.getElementById('initial-chart');
    if (!ctx) {
        console.error("Element #initial-chart not found!");
        return;
    }
    
    // Destroy existing chart
    if (window.initialChartInstance) {
        window.initialChartInstance.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    // Survival by Gender
    const male = rawData.train.filter(d => d.Sex === 'male');
    const female = rawData.train.filter(d => d.Sex === 'female');
    
    const maleSurvival = male.length > 0 ? 
        (male.filter(d => d.Survived === 1).length / male.length * 100).toFixed(1) : 0;
    const femaleSurvival = female.length > 0 ? 
        (female.filter(d => d.Survived === 1).length / female.length * 100).toFixed(1) : 0;
    
    window.initialChartInstance = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['Male', 'Female'],
            datasets: [{
                label: 'Survival Rate (%)',
                data: [parseFloat(maleSurvival), parseFloat(femaleSurvival)],
                backgroundColor: ['#3b82f6', '#ec4899'],
                borderColor: ['#2563eb', '#db2777'],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Survival Rate (%)',
                        font: { weight: 'bold' }
                    },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Survival: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

// ==================== 2. PREPROCESSING ====================
function preprocessData() {
    console.log("Preprocessing data...");
    
    if (!rawData.train || !rawData.test) {
        showStatus('❌ Please load data first', 'error', 'preprocess-status');
        return;
    }
    
    showStatus('🔧 Preprocessing data...', 'info', 'preprocess-status');
    
    try {
        // Get preprocessing options
        const addFamilySize = document.getElementById('add-family-size')?.checked || true;
        const addIsAlone = document.getElementById('add-is-alone')?.checked || true;
        
        // Impute missing values
        const trainProcessed = imputeMissingValues(rawData.train, rawData.train);
        const testProcessed = imputeMissingValues(rawData.test, rawData.train);
        
        // Prepare features
        const { features: X_train, target: y_train, featureNames } = prepareFeatures(
            trainProcessed, 
            true, 
            addFamilySize, 
            addIsAlone
        );
        
        const { features: X_test } = prepareFeatures(
            testProcessed, 
            false, 
            addFamilySize, 
            addIsAlone
        );
        
        // Store processed data
        processedData.X_train = X_train;
        processedData.y_train = y_train;
        processedData.X_test = X_test;
        processedData.featureNames = featureNames;
        
        console.log("Preprocessing complete!");
        console.log("Training features shape:", X_train.length, "x", X_train[0]?.length);
        console.log("Test features shape:", X_test.length, "x", X_test[0]?.length);
        console.log("Feature names:", featureNames);
        
        showStatus(`✅ Preprocessing complete! ${featureNames.length} features created`, 'success', 'preprocess-status');
        
        // Display feature information
        const featureInfo = document.getElementById('feature-info');
        const featureList = document.getElementById('feature-list');
        
        if (featureInfo) featureInfo.style.display = 'block';
        if (featureList) {
            featureList.innerHTML = `
                <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #e5e7eb;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <p><strong>Total Features:</strong> ${featureNames.length}</p>
                            <p><strong>Training Samples:</strong> ${X_train.length}</p>
                            <p><strong>Test Samples:</strong> ${X_test.length}</p>
                        </div>
                        <div>
                            <p><strong>Survived:</strong> ${y_train.filter(y => y === 1).length} samples</p>
                            <p><strong>Not Survived:</strong> ${y_train.filter(y => y === 0).length} samples</p>
                        </div>
                    </div>
                    <div style="margin-top: 10px;">
                        <strong>Feature Names (first 10):</strong>
                        <div style="font-size: 0.9rem; color: #6b7280; margin-top: 5px; font-family: monospace;">
                            ${featureNames.slice(0, 10).join(', ')}${featureNames.length > 10 ? '...' : ''}
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Enable model controls
        const modelControls = document.getElementById('model-controls');
        if (modelControls) {
            modelControls.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error preprocessing:', error);
        showStatus(`❌ Preprocessing error: ${error.message}`, 'error', 'preprocess-status');
    }
}

function imputeMissingValues(data, referenceData = null) {
    const ref = referenceData || data;
    
    // Calculate medians
    const ageValues = ref.filter(r => r.Age && !isNaN(r.Age)).map(r => r.Age);
    const fareValues = ref.filter(r => r.Fare && !isNaN(r.Fare)).map(r => r.Fare);
    
    const ageMedian = ageValues.length > 0 ? 
        ageValues.sort((a, b) => a - b)[Math.floor(ageValues.length / 2)] : 29.7;
    
    const fareMedian = fareValues.length > 0 ? 
        fareValues.sort((a, b) => a - b)[Math.floor(fareValues.length / 2)] : 14.45;
    
    // Mode for Embarked
    const embarkedCounts = {};
    ref.filter(r => r.Embarked).forEach(r => {
        embarkedCounts[r.Embarked] = (embarkedCounts[r.Embarked] || 0) + 1;
    });
    const embarkedMode = Object.keys(embarkedCounts).length > 0 ? 
        Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
    
    // Impute values
    return data.map(row => ({
        ...row,
        Age: (row.Age && !isNaN(row.Age)) ? row.Age : ageMedian,
        Fare: (row.Fare && !isNaN(row.Fare)) ? row.Fare : fareMedian,
        Embarked: row.Embarked || embarkedMode,
        Cabin: row.Cabin || 'Unknown'
    }));
}

function prepareFeatures(data, isTraining, addFamilySize = true, addIsAlone = true) {
    const features = [];
    const target = [];
    const featureNames = [];
    
    // Process each row
    data.forEach(row => {
        const featureVector = [];
        
        // One-hot encode Pclass
        const pclass = row.Pclass || 3;
        for (let i = 1; i <= 3; i++) {
            featureVector.push(pclass === i ? 1 : 0);
            if (featureNames.length < 3) featureNames.push(`Pclass_${i}`);
        }
        
        // One-hot encode Sex
        const sex = row.Sex || 'male';
        featureVector.push(sex === 'female' ? 1 : 0);
        featureVector.push(sex === 'male' ? 1 : 0);
        if (featureNames.length < 5) {
            featureNames.push('Sex_female');
            featureNames.push('Sex_male');
        }
        
        // Age (standardized)
        const age = row.Age || 29.7;
        const ageStd = (age - 29.7) / 13.5;
        featureVector.push(ageStd);
        if (featureNames.length < 6) featureNames.push('Age_std');
        
        // SibSp and Parch
        featureVector.push(row.SibSp || 0);
        featureVector.push(row.Parch || 0);
        if (featureNames.length < 8) {
            featureNames.push('SibSp');
            featureNames.push('Parch');
        }
        
        // Fare (standardized)
        const fare = row.Fare || 14.45;
        const fareStd = (fare - 32.2) / 49.7;
        featureVector.push(fareStd);
        if (featureNames.length < 9) featureNames.push('Fare_std');
        
        // One-hot encode Embarked
        const embarked = row.Embarked || 'S';
        ['C', 'Q', 'S'].forEach(port => {
            featureVector.push(embarked === port ? 1 : 0);
            if (featureNames.length < 12) featureNames.push(`Embarked_${port}`);
        });
        
        // Feature engineering
        if (addFamilySize) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            featureVector.push(familySize);
            if (featureNames.length < 13) featureNames.push('FamilySize');
        }
        
        if (addIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureVector.push(isAlone);
            if (featureNames.length < 14) featureNames.push('IsAlone');
        }
        
        features.push(featureVector);
        
        if (isTraining) {
            target.push(row.Survived !== undefined ? row.Survived : 0);
        }
    });
    
    return { features, target, featureNames };
}

// ==================== 3. MODEL BUILDING ====================
function buildModel() {
    console.log("Building model...");
    
    if (!processedData.X_train || !processedData.y_train) {
        showStatus('❌ Please preprocess data first', 'error', 'model-status');
        return;
    }
    
    showStatus('🏗️ Building neural network...', 'info', 'model-status');
    
    try {
        // Get model parameters
        const hiddenUnits = parseInt(document.getElementById('hidden-units')?.value || 16);
        const learningRate = parseFloat(document.getElementById('learning-rate')?.value || 0.001);
        const activation = document.getElementById('activation')?.value || 'relu';
        
        // Create model
        model = tf.sequential();
        
        // Input layer
        model.add(tf.layers.dense({
            units: hiddenUnits,
            activation: activation,
            inputShape: [processedData.featureNames.length]
        }));
        
        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile model
        model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        showStatus(`✅ Model built successfully! (${hiddenUnits} hidden units, ${activation} activation)`, 'success', 'model-status');
        
        // Enable training controls
        const trainingControls = document.getElementById('training-controls');
        if (trainingControls) {
            trainingControls.style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error building model:', error);
        showStatus(`❌ Model building error: ${error.message}`, 'error', 'model-status');
    }
}

function displayModelSummary() {
    const summaryElement = document.getElementById('model-summary');
    if (!summaryElement) return;
    
    let summaryHTML = `
        <div style="color: #3b82f6; font-weight: bold; margin-bottom: 10px;">
            <i class="fas fa-layer-group"></i> Model Architecture
        </div>
    `;
    
    // Layer information
    model.layers.forEach((layer, i) => {
        const config = layer.getConfig();
        
        summaryHTML += `
            <div style="margin-bottom: 8px; padding: 8px; background: #f8fafc; border-radius: 6px;">
                <div style="font-weight: bold; color: #1f2937;">
                    ${i === 0 ? 'Hidden Layer' : 'Output Layer'} (${layer.getClassName()})
                </div>
                <div style="font-size: 0.9rem; color: #6b7280;">
                    Units: ${config.units} | Activation: ${config.activation}
                </div>
            </div>
        `;
    });
    
    // Calculate total parameters
    const totalParams = model.countParams();
    summaryHTML += `
        <div style="margin-top: 15px; padding: 10px; background: linear-gradient(135deg, #3b82f6, #1d4ed8); 
                    color: white; border-radius: 6px; font-weight: bold;">
            <i class="fas fa-calculator"></i> Total Parameters: ${totalParams.toLocaleString()}
        </div>
    `;
    
    summaryElement.innerHTML = summaryHTML;
}

// ==================== HELPER FUNCTIONS ====================
function resetAll() {
    console.log("Resetting all...");
    
    rawData = { train: null, test: null };
    processedData = { X_train: null, y_train: null, X_test: null, featureNames: [], scalers: {} };
    model = null;
    trainingHistory = [];
    predictions = null;
    validationData = null;
    currentThreshold = 0.5;
    isTraining = false;
    
    // Reset file inputs
    const trainFile = document.getElementById('train-file');
    const testFile = document.getElementById('test-file');
    if (trainFile) trainFile.value = '';
    if (testFile) testFile.value = '';
    
    // Reset UI elements
    const elementsToHide = ['data-overview', 'preprocess-controls', 'feature-info', 
                          'model-controls', 'training-controls', 'metrics-display', 'prediction-controls'];
    
    elementsToHide.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.style.display = 'none';
    });
    
    // Reset charts
    if (window.initialChartInstance) {
        window.initialChartInstance.destroy();
        window.initialChartInstance = null;
    }
    
    if (trainingChart) {
        trainingChart.destroy();
        trainingChart = null;
    }
    
    if (rocChart) {
        rocChart.destroy();
        rocChart = null;
    }
    
    if (confusionMatrixChart) {
        confusionMatrixChart.destroy();
        confusionMatrixChart = null;
    }
    
    // Reset status
    showStatus('🔄 All data and models have been reset. Ready for new data.', 'info');
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==================== INITIALIZATION ====================
// Make functions available globally
window.loadData = loadData;
window.resetAll = resetAll;
window.preprocessData = preprocessData;
window.buildModel = buildModel;
window.startTraining = startTraining;
window.stopTraining = stopTraining;
window.scrollToTop = scrollToTop;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log("Titanic ML Explorer initialized!");
    showStatus('✅ System ready. Upload train.csv and test.csv files to begin.', 'success');
    
    // Setup drag and drop
    setupDragAndDrop();
});

function setupDragAndDrop() {
    const uploadZone = document.querySelector('.upload-zone');
    if (!uploadZone) return;
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#3b82f6';
        uploadZone.style.background = 'linear-gradient(to bottom, #f0f9ff, #e0f2fe)';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = '#e5e7eb';
        uploadZone.style.background = 'linear-gradient(to bottom, #f8fafc, #f1f5f9)';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#e5e7eb';
        uploadZone.style.background = 'linear-gradient(to bottom, #f8fafc, #f1f5f9)';
        
        const files = e.dataTransfer.files;
        handleDroppedFiles(files);
    });
}

function handleDroppedFiles(files) {
    const trainFileInput = document.getElementById('train-file');
    const testFileInput = document.getElementById('test-file');
    
    Array.from(files).forEach(file => {
        if (file.name.toLowerCase().includes('train')) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            trainFileInput.files = dataTransfer.files;
            showStatus(`✅ Added train file: ${file.name}`, 'success');
        } else if (file.name.toLowerCase().includes('test')) {
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            testFileInput.files = dataTransfer.files;
            showStatus(`✅ Added test file: ${file.name}`, 'success');
        }
    });
}