// app.js - Titanic Survival Classifier with TensorFlow.js
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
    if (!element) return;
    
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

async function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

async function parseCSV(file) {
    return new Promise((resolve, reject) => {
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                console.log(`Parsed ${results.data.length} rows from ${file.name}`);
                resolve(results.data);
            },
            error: (error) => reject(error)
        });
    });
}

// ==================== 1. DATA LOADING & EXPLORATION ====================
async function loadData() {
    const trainFile = document.getElementById('train-file');
    const testFile = document.getElementById('test-file');
    
    if (!trainFile.files[0] || !testFile.files[0]) {
        showStatus('❌ Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    showStatus('⏳ Loading CSV files...', 'info');
    
    try {
        // Parse CSV files
        const trainText = await readFile(trainFile.files[0]);
        const testText = await readFile(testFile.files[0]);
        
        const trainResults = await parseCSV(trainFile.files[0]);
        const testResults = await parseCSV(testFile.files[0]);
        
        rawData.train = trainResults;
        rawData.test = testResults;
        
        showStatus(`✅ Loaded ${rawData.train.length} training and ${rawData.test.length} test samples`, 'success');
        
        // Update UI
        updateDataOverview();
        createInitialChart();
        
        // Enable preprocessing controls
        document.getElementById('preprocess-controls').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

function updateDataOverview() {
    // Update statistics
    document.getElementById('total-passengers').textContent = rawData.train.length + rawData.test.length;
    document.getElementById('train-count').textContent = rawData.train.length;
    document.getElementById('test-count').textContent = rawData.test.length;
    
    // Calculate survival rate
    const survived = rawData.train.filter(row => row.Survived === 1).length;
    const survivalRate = ((survived / rawData.train.length) * 100).toFixed(1);
    document.getElementById('survival-rate').textContent = `${survivalRate}%`;
    
    // Show data preview
    showDataPreview();
    
    // Analyze missing values
    showMissingValues();
    
    // Show the overview section
    document.getElementById('data-overview').style.display = 'block';
    
    // Show insights
    const insights = document.getElementById('initial-insights');
    if (insights) {
        insights.innerHTML = `
            <div class="insight-item">Survival rate: ${survivalRate}% (${survived} out of ${rawData.train.length})</div>
            <div class="insight-item">Women survival: ${calculateSurvivalBySex('female')}%</div>
            <div class="insight-item">Men survival: ${calculateSurvivalBySex('male')}%</div>
            <div class="insight-item">1st class survival: ${calculateSurvivalByClass(1)}%</div>
        `;
    }
}

function showDataPreview() {
    const tbody = document.getElementById('preview-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    // Show first 5 rows from training data
    rawData.train.slice(0, 5).forEach(row => {
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
    if (!container) return;
    
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

function calculateSurvivalBySex(sex) {
    const passengers = rawData.train.filter(row => row.Sex === sex);
    const survived = passengers.filter(row => row.Survived === 1).length;
    return passengers.length > 0 ? ((survived / passengers.length) * 100).toFixed(1) : '0';
}

function calculateSurvivalByClass(pclass) {
    const passengers = rawData.train.filter(row => row.Pclass === pclass);
    const survived = passengers.filter(row => row.Survived === 1).length;
    return passengers.length > 0 ? ((survived / passengers.length) * 100).toFixed(1) : '0';
}

function createInitialChart() {
    const ctx = document.getElementById('initial-chart');
    if (!ctx) return;
    
    const chartCtx = ctx.getContext('2d');
    
    // Destroy existing chart
    if (window.initialChart) {
        window.initialChart.destroy();
    }
    
    // Survival by Gender
    const male = rawData.train.filter(d => d.Sex === 'male');
    const female = rawData.train.filter(d => d.Sex === 'female');
    
    const maleSurvival = (male.filter(d => d.Survived === 1).length / male.length * 100).toFixed(1);
    const femaleSurvival = (female.filter(d => d.Survived === 1).length / female.length * 100).toFixed(1);
    
    window.initialChart = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['Male', 'Female'],
            datasets: [{
                label: 'Survival Rate (%)',
                data: [maleSurvival, femaleSurvival],
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
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
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
async function preprocessData() {
    showStatus('🔧 Preprocessing data...', 'info', 'preprocess-status');
    
    try {
        // Get preprocessing options
        const addFamilySize = document.getElementById('add-family-size')?.checked || true;
        const addIsAlone = document.getElementById('add-is-alone')?.checked || true;
        
        // Process training data
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
        
        showStatus(`✅ Preprocessing complete! ${featureNames.length} features created`, 'success', 'preprocess-status');
        
        // Display feature information
        document.getElementById('feature-info').style.display = 'block';
        document.getElementById('feature-list').innerHTML = `
            <div style="background: #f8fafc; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #e5e7eb;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <p><strong>Total Features:</strong> ${featureNames.length}</p>
                        <p><strong>Training Samples:</strong> ${X_train.length}</p>
                        <p><strong>Test Samples:</strong> ${X_test.length}</p>
                    </div>
                    <div>
                        <p><strong>Target Values:</strong> ${y_train.filter(y => y === 1).length} survived, ${y_train.filter(y => y === 0).length} not survived</p>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Feature Names:</strong>
                    <div style="font-size: 0.9rem; color: #6b7280; margin-top: 5px;">
                        ${featureNames.slice(0, 10).join(', ')}${featureNames.length > 10 ? '...' : ''}
                    </div>
                </div>
            </div>
        `;
        
        // Enable model controls
        document.getElementById('model-controls').style.display = 'block';
        
    } catch (error) {
        console.error('Error preprocessing:', error);
        showStatus(`❌ Preprocessing error: ${error.message}`, 'error', 'preprocess-status');
    }
}

function imputeMissingValues(data, referenceData = null) {
    const ref = referenceData || data;
    
    // Calculate medians and modes from reference data
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
    const embarkedMode = Object.keys(embarkedCounts).reduce((a, b) => 
        embarkedCounts[a] > embarkedCounts[b] ? a : b, 'S'
    );
    
    // Impute values
    return data.map(row => ({
        ...row,
        Age: (row.Age && !isNaN(row.Age)) ? row.Age : ageMedian,
        Fare: (row.Fare && !isNaN(row.Fare)) ? row.Fare : fareMedian,
        Embarked: row.Embarked || embarkedMode,
        Cabin: row.Cabin || 'Unknown',
        SibSp: row.SibSp || 0,
        Parch: row.Parch || 0
    }));
}

function prepareFeatures(data, isTraining, addFamilySize = true, addIsAlone = true) {
    const features = [];
    const target = [];
    const featureNames = [];
    
    // Process each row
    data.forEach(row => {
        const featureVector = [];
        
        // One-hot encode Pclass (1, 2, 3)
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
        const ageStd = (age - 29.7) / 13.5; // Approximate standardization
        featureVector.push(ageStd);
        if (featureNames.length < 6) featureNames.push('Age_std');
        
        // SibSp and Parch (as is)
        featureVector.push(row.SibSp || 0);
        featureVector.push(row.Parch || 0);
        if (featureNames.length < 8) {
            featureNames.push('SibSp');
            featureNames.push('Parch');
        }
        
        // Fare (standardized)
        const fare = row.Fare || 14.45;
        const fareStd = (fare - 32.2) / 49.7; // Approximate standardization
        featureVector.push(fareStd);
        if (featureNames.length < 9) featureNames.push('Fare_std');
        
        // One-hot encode Embarked (C, Q, S)
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
        
        // Add to features array
        features.push(featureVector);
        
        // Add target if training data
        if (isTraining) {
            target.push(row.Survived !== undefined ? row.Survived : 0);
        }
    });
    
    return { features, target, featureNames };
}

// ==================== 3. MODEL BUILDING ====================
async function buildModel() {
    showStatus('🏗️ Building neural network...', 'info', 'model-status');
    
    try {
        // Get model parameters
        const hiddenUnits = parseInt(document.getElementById('hidden-units')?.value || 16);
        const learningRate = parseFloat(document.getElementById('learning-rate')?.value || 0.001);
        const activation = document.getElementById('activation')?.value || 'relu';
        
        // Create sequential model
        model = tf.sequential();
        
        // Input layer (automatically inferred)
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
        document.getElementById('training-controls').style.display = 'block';
        
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
        const layerType = layer.getClassName();
        const config = layer.getConfig();
        
        summaryHTML += `
            <div style="margin-bottom: 8px; padding: 8px; background: #f8fafc; border-radius: 6px;">
                <div style="font-weight: bold; color: #1f2937;">
                    ${i === 0 ? 'Hidden Layer' : 'Output Layer'} (${layerType})
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

// ==================== 4. TRAINING ====================
async function startTraining() {
    if (!model || !processedData.X_train || !processedData.y_train) {
        showStatus('❌ Please build model and preprocess data first', 'error', 'training-status');
        return;
    }
    
    // Get training parameters
    const epochs = parseInt(document.getElementById('epochs')?.value || 50);
    const batchSize = parseInt(document.getElementById('batch-size')?.value || 32);
    const validationSplit = parseFloat(document.getElementById('validation-split')?.value || 0.2);
    
    showStatus('🏋️‍♂️ Starting model training...', 'info', 'training-status');
    isTraining = true;
    
    // Convert arrays to tensors
    const xs = tf.tensor2d(processedData.X_train);
    const ys = tf.tensor2d(processedData.y_train, [processedData.y_train.length, 1]);
    
    // Setup training progress
    const progressBar = document.getElementById('training-progress');
    const trainingStats = document.getElementById('training-stats');
    
    try {
        // Train the model
        const history = await model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: validationSplit,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // Update progress bar
                    const progress = ((epoch + 1) / epochs) * 100;
                    progressBar.style.width = `${progress}%`;
                    
                    // Update training stats
                    trainingStats.innerHTML = `
                        <div style="font-size: 1.2rem; font-weight: bold; color: #3b82f6;">
                            Epoch ${epoch + 1}/${epochs}
                        </div>
                        <div style="color: #6b7280; margin-top: 5px;">
                            Loss: ${logs.loss.toFixed(4)} | Accuracy: ${(logs.acc * 100).toFixed(1)}%
                        </div>
                    `;
                    
                    // Store history for plotting
                    trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        valLoss: logs.val_loss,
                        valAccuracy: logs.val_acc
                    });
                    
                    // Update training chart
                    updateTrainingChart();
                },
                onTrainEnd: () => {
                    isTraining = false;
                    showStatus('✅ Training completed successfully!', 'success', 'training-status');
                    
                    // Enable metrics display
                    document.getElementById('metrics-display').style.display = 'block';
                    
                    // Make predictions on validation set
                    evaluateModel();
                }
            }
        });
        
        // Clean up tensors
        xs.dispose();
        ys.dispose();
        
    } catch (error) {
        console.error('Error during training:', error);
        showStatus(`❌ Training error: ${error.message}`, 'error', 'training-status');
        isTraining = false;
    }
}

function stopTraining() {
    isTraining = false;
    showStatus('⏹️ Training stopped by user', 'info', 'training-status');
}

function updateTrainingChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx || trainingHistory.length === 0) return;
    
    const chartCtx = ctx.getContext('2d');
    
    // Destroy existing chart
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    // Prepare data
    const epochs = trainingHistory.map(h => h.epoch);
    const losses = trainingHistory.map(h => h.loss);
    const valLosses = trainingHistory.map(h => h.valLoss);
    const accuracies = trainingHistory.map(h => h.accuracy);
    const valAccuracies = trainingHistory.map(h => h.valAccuracy);
    
    trainingChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: 'Training Loss',
                    data: losses,
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Validation Loss',
                    data: valLosses,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y'
                },
                {
                    label: 'Training Accuracy',
                    data: accuracies,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Validation Accuracy',
                    data: valAccuracies,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Epoch',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        drawOnChartArea: false,
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label.includes('Accuracy')) {
                                return `${label}: ${(context.raw * 100).toFixed(2)}%`;
                            }
                            return `${label}: ${context.raw.toFixed(4)}`;
                        }
                    }
                }
            }
        }
    });
}

// ==================== 5. EVALUATION & METRICS ====================
async function evaluateModel() {
    if (!model || !processedData.X_train || !processedData.y_train) {
        showStatus('❌ Model not trained yet', 'error', 'metrics-status');
        return;
    }
    
    showStatus('📊 Evaluating model performance...', 'info', 'metrics-status');
    
    try {
        // Split data for validation
        const validationSize = Math.floor(processedData.X_train.length * 0.2);
        const X_val = processedData.X_train.slice(0, validationSize);
        const y_val = processedData.y_train.slice(0, validationSize);
        
        // Convert to tensors
        const xsVal = tf.tensor2d(X_val);
        
        // Make predictions
        const predictions = model.predict(xsVal);
        const probs = await predictions.data();
        predictions.dispose();
        xsVal.dispose();
        
        // Store for ROC curve
        validationData = {
            labels: y_val,
            probabilities: Array.from(probs)
        };
        
        // Calculate metrics
        const { accuracy, loss } = calculateMetrics(y_val, probs);
        const auc = calculateAUC(y_val, probs);
        
        // Update UI
        document.getElementById('accuracy-value').textContent = accuracy.toFixed(3);
        document.getElementById('loss-value').textContent = loss.toFixed(4);
        document.getElementById('auc-value').textContent = auc.toFixed(3);
        
        // Create confusion matrix
        createConfusionMatrix(y_val, probs);
        
        // Create ROC curve
        createROCCurve(y_val, probs);
        
        // Update threshold metrics
        updateThresholdMetrics();
        
        showStatus('✅ Model evaluation complete!', 'success', 'metrics-status');
        
        // Enable prediction controls
        document.getElementById('prediction-controls').style.display = 'block';
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        showStatus(`❌ Evaluation error: ${error.message}`, 'error', 'metrics-status');
    }
}

function calculateMetrics(trueLabels, predictedProbs) {
    const predictedLabels = predictedProbs.map(p => p >= currentThreshold ? 1 : 0);
    
    // Calculate accuracy
    let correct = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === predictedLabels[i]) {
            correct++;
        }
    }
    const accuracy = correct / trueLabels.length;
    
    // Calculate binary crossentropy loss
    let loss = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        const y = trueLabels[i];
        const p = predictedProbs[i];
        const epsilon = 1e-15; // Prevent log(0)
        const p_clipped = Math.max(epsilon, Math.min(1 - epsilon, p));
        loss += y * Math.log(p_clipped) + (1 - y) * Math.log(1 - p_clipped);
    }
    loss = -loss / trueLabels.length;
    
    return { accuracy, loss };
}

function calculateAUC(trueLabels, predictedProbs) {
    // Simple AUC calculation (trapezoidal rule)
    const sorted = trueLabels.map((label, i) => ({
        label,
        prob: predictedProbs[i]
    })).sort((a, b) => b.prob - a.prob);
    
    let truePositives = 0;
    let falsePositives = 0;
    let prevTPR = 0;
    let prevFPR = 0;
    let auc = 0;
    
    const totalPositives = trueLabels.filter(l => l === 1).length;
    const totalNegatives = trueLabels.filter(l => l === 0).length;
    
    for (const { label, prob } of sorted) {
        if (label === 1) {
            truePositives++;
        } else {
            falsePositives++;
        }
        
        const tpr = truePositives / totalPositives;
        const fpr = falsePositives / totalNegatives;
        
        auc += (fpr - prevFPR) * (tpr + prevTPR) / 2;
        prevTPR = tpr;
        prevFPR = fpr;
    }
    
    return auc;
}

function createConfusionMatrix(trueLabels, predictedProbs) {
    const ctx = document.getElementById('confusion-matrix-chart');
    if (!ctx) return;
    
    const chartCtx = ctx.getContext('2d');
    
    // Calculate confusion matrix
    const predictedLabels = predictedProbs.map(p => p >= currentThreshold ? 1 : 0);
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predictedLabels[i] === 1) tp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 1) fp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 0) tn++;
        if (trueLabels[i] === 1 && predictedLabels[i] === 0) fn++;
    }
    
    // Destroy existing chart
    if (confusionMatrixChart) {
        confusionMatrixChart.destroy();
    }
    
    confusionMatrixChart = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
            datasets: [{
                label: 'Count',
                data: [tp, fp, tn, fn],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(239, 68, 68, 0.7)',
                    'rgba(16, 185, 129, 0.7)',
                    'rgba(239, 68, 68, 0.7)'
                ],
                borderColor: [
                    '#10b981',
                    '#ef4444',
                    '#10b981',
                    '#ef4444'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count',
                        font: {
                            weight: 'bold'
                        }
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function createROCCurve(trueLabels, predictedProbs) {
    const ctx = document.getElementById('roc-chart');
    if (!ctx) return;
    
    const chartCtx = ctx.getContext('2d');
    
    // Generate ROC curve points
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocPoints = thresholds.map(threshold => {
        const predictedLabels = predictedProbs.map(p => p >= threshold ? 1 : 0);
        
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (let i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] === 1 && predictedLabels[i] === 1) tp++;
            if (trueLabels[i] === 0 && predictedLabels[i] === 1) fp++;
            if (trueLabels[i] === 0 && predictedLabels[i] === 0) tn++;
            if (trueLabels[i] === 1 && predictedLabels[i] === 0) fn++;
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        return { fpr, tpr, threshold };
    });
    
    // Destroy existing chart
    if (rocChart) {
        rocChart.destroy();
    }
    
    rocChart = new Chart(chartCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'ROC Curve',
                data: rocPoints.map(p => ({ x: p.fpr, y: p.tpr })),
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                pointRadius: 0
            }, {
                label: 'Random Classifier',
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: '#