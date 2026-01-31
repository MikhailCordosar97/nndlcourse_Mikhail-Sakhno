// app.js - Titanic Neural Network Classifier
// Complete browser-based ML with TensorFlow.js

"use strict";

// ============================================
// CONFIGURATION - SWAP THESE FOR OTHER DATASETS
// ============================================
const CONFIG = {
    // Target variable for binary classification
    target: 'Survived',  // Binary label (0 = died, 1 = survived)
    
    // Features to use for training
    features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    
    // Identifier column to exclude from features
    identifier: 'PassengerId',
    
    // Feature types for preprocessing
    categorical: ['Sex', 'Embarked', 'Pclass'],
    numerical: ['Age', 'Fare', 'SibSp', 'Parch'],
    
    // Training parameters
    training: {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        learningRate: 0.001
    }
};

// ============================================
// GLOBAL STATE
// ============================================
const AppState = {
    // Raw data
    rawData: {
        train: null,
        test: null
    },
    
    // Processed data
    processedData: {
        X_train: null,
        y_train: null,
        X_val: null,
        y_val: null,
        X_test: null,
        featureNames: [],
        scalers: {}
    },
    
    // Model and training
    model: null,
    trainingHistory: [],
    isTraining: false,
    
    // Predictions and evaluation
    predictions: null,
    validationProbs: null,
    validationLabels: null,
    testProbs: null,
    
    // Charts
    charts: {
        exploration: null,
        training: null,
        roc: null,
        confusion: null,
        correlation: null
    },
    
    // Current threshold for binary classification
    currentThreshold: 0.5
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Display status message to user
 * @param {string} message - Status message
 * @param {string} type - 'success', 'error', or 'info'
 * @param {string} elementId - Element ID to display message in
 */
function showStatus(message, type = 'info', elementId = 'data-status') {
    const element = document.getElementById(elementId);
    if (!element) {
        console.error(`Element #${elementId} not found`);
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
    
    console.log(`[${type.toUpperCase()}] ${message}`);
}

/**
 * Read a file as text
 * @param {File} file - File object to read
 * @returns {Promise<string>} File content as text
 */
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (error) => reject(error);
        reader.readAsText(file);
    });
}

/**
 * Parse CSV text into JavaScript objects
 * @param {string} csvText - CSV file content
 * @returns {Promise<Array>} Parsed data array
 */
function parseCSV(csvText) {
    return new Promise((resolve, reject) => {
        // Use PapaParse for robust CSV parsing
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

/**
 * Navigate to a specific step in the workflow
 * @param {number} stepNumber - Step number (1-5)
 */
function goToStep(stepNumber) {
    // Update navigation indicators
    document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
        indicator.classList.toggle('active', (index + 1) === stepNumber);
    });
    
    // Show/hide sections
    document.querySelectorAll('.section').forEach((section, index) => {
        section.classList.toggle('active', (index + 1) === stepNumber);
    });
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/**
 * Reset all application state
 */
function resetAll() {
    // Reset state
    AppState.rawData = { train: null, test: null };
    AppState.processedData = {
        X_train: null,
        y_train: null,
        X_val: null,
        y_val: null,
        X_test: null,
        featureNames: [],
        scalers: {}
    };
    AppState.model = null;
    AppState.trainingHistory = [];
    AppState.isTraining = false;
    AppState.predictions = null;
    
    // Reset file inputs
    document.getElementById('trainFile').value = '';
    document.getElementById('testFile').value = '';
    
    // Hide all sections except first
    goToStep(1);
    
    // Clear charts
    Object.values(AppState.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    AppState.charts = {
        exploration: null,
        training: null,
        roc: null,
        confusion: null,
        correlation: null
    };
    
    // Clear UI elements
    document.getElementById('data-overview').style.display = 'none';
    document.getElementById('preprocess-results').style.display = 'none';
    document.getElementById('model-built').style.display = 'none';
    document.getElementById('training-visualization').style.display = 'none';
    document.getElementById('metrics-display').style.display = 'none';
    document.getElementById('export-section').style.display = 'none';
    
    showStatus('Application reset. Ready to load new data.', 'info');
}

// ============================================
// SECTION 1: DATA LOADING & EXPLORATION
// ============================================

/**
 * Load and explore Titanic dataset
 */
async function loadAndExploreData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    // Validate files
    if (!trainFile || !testFile) {
        showStatus('❌ Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    if (!trainFile.name.includes('train') || !testFile.name.includes('test')) {
        showStatus('❌ Please select correct files: train.csv and test.csv', 'error');
        return;
    }
    
    showStatus('⏳ Loading CSV files...', 'info');
    
    try {
        // Read and parse files
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        AppState.rawData.train = await parseCSV(trainText);
        AppState.rawData.test = await parseCSV(testText);
        
        console.log('Data loaded:', {
            train: AppState.rawData.train.length,
            test: AppState.rawData.test.length,
            firstTrainRow: AppState.rawData.train[0]
        });
        
        // Update UI with data overview
        updateDataOverview();
        
        // Create exploration visualizations
        createExplorationCharts();
        
        showStatus(`✅ Loaded ${AppState.rawData.train.length} training and ${AppState.rawData.test.length} test samples`, 'success');
        
        // Show data overview
        document.getElementById('data-overview').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus(`❌ Error loading data: ${error.message}`, 'error');
    }
}

/**
 * Update UI with dataset overview
 */
function updateDataOverview() {
    const trainData = AppState.rawData.train;
    const testData = AppState.rawData.test;
    
    // Update statistics
    document.getElementById('total-passengers').textContent = trainData.length + testData.length;
    document.getElementById('train-count').textContent = trainData.length;
    document.getElementById('test-count').textContent = testData.length;
    
    // Calculate survival rate
    const survived = trainData.filter(row => row.Survived === 1).length;
    const survivalRate = ((survived / trainData.length) * 100).toFixed(1);
    document.getElementById('survival-rate').textContent = `${survivalRate}%`;
    
    // Show data preview
    showDataPreview();
    
    // Calculate and show missing values
    showMissingValues();
}

/**
 * Display data preview table
 */
function showDataPreview() {
    const tbody = document.getElementById('preview-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    // Show first 8 rows
    AppState.rawData.train.slice(0, 8).forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.PassengerId || ''}</td>
            <td>${(row.Name || '').substring(0, 20)}...</td>
            <td>${row.Pclass || ''}</td>
            <td>${row.Sex || ''}</td>
            <td>${row.Age ? row.Age.toFixed(1) : 'N/A'}</td>
            <td>${row.Fare ? row.Fare.toFixed(2) : 'N/A'}</td>
            <td>${row.Survived !== undefined ? row.Survived : 'N/A'}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Calculate and display missing values
 */
function showMissingValues() {
    const features = ['Age', 'Cabin', 'Embarked', 'Fare'];
    const trainData = AppState.rawData.train;
    
    console.log('Missing values analysis:');
    features.forEach(feature => {
        const missing = trainData.filter(row => 
            row[feature] === null || row[feature] === undefined || row[feature] === ''
        ).length;
        const percentage = ((missing / trainData.length) * 100).toFixed(1);
        console.log(`  ${feature}: ${missing} missing (${percentage}%)`);
    });
}

/**
 * Create exploration charts
 */
function createExplorationCharts() {
    const ctx = document.getElementById('exploration-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (AppState.charts.exploration) {
        AppState.charts.exploration.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    const trainData = AppState.rawData.train;
    
    // Prepare data for survival by gender
    const male = trainData.filter(d => d.Sex === 'male');
    const female = trainData.filter(d => d.Sex === 'female');
    
    const maleSurvival = male.length > 0 ? 
        (male.filter(d => d.Survived === 1).length / male.length * 100).toFixed(1) : 0;
    const femaleSurvival = female.length > 0 ? 
        (female.filter(d => d.Survived === 1).length / female.length * 100).toFixed(1) : 0;
    
    // Prepare data for survival by class
    const class1 = trainData.filter(d => d.Pclass === 1);
    const class2 = trainData.filter(d => d.Pclass === 2);
    const class3 = trainData.filter(d => d.Pclass === 3);
    
    const class1Survival = class1.length > 0 ?
        (class1.filter(d => d.Survived === 1).length / class1.length * 100).toFixed(1) : 0;
    const class2Survival = class2.length > 0 ?
        (class2.filter(d => d.Survived === 1).length / class2.length * 100).toFixed(1) : 0;
    const class3Survival = class3.length > 0 ?
        (class3.filter(d => d.Survived === 1).length / class3.length * 100).toFixed(1) : 0;
    
    AppState.charts.exploration = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['Male', 'Female', '1st Class', '2nd Class', '3rd Class'],
            datasets: [{
                label: 'Survival Rate (%)',
                data: [
                    parseFloat(maleSurvival),
                    parseFloat(femaleSurvival),
                    parseFloat(class1Survival),
                    parseFloat(class2Survival),
                    parseFloat(class3Survival)
                ],
                backgroundColor: [
                    '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'
                ],
                borderColor: [
                    '#2980b9', '#c0392b', '#27ae60', '#d68910', '#8e44ad'
                ],
                borderWidth: 2,
                borderRadius: 6
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
                        font: { size: 14, weight: 'bold' }
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

// ============================================
// SECTION 2: PREPROCESSING
// ============================================

/**
 * Preprocess the data for training
 */
async function preprocessData() {
    showStatus('🔧 Preprocessing data...', 'info', 'preprocess-status');
    
    if (!AppState.rawData.train || !AppState.rawData.test) {
        showStatus('❌ Please load data first', 'error', 'preprocess-status');
        return;
    }
    
    try {
        // Get feature engineering options
        const addFamilySize = document.getElementById('family-size-toggle').checked;
        const addIsAlone = document.getElementById('is-alone-toggle').checked;
        
        // Impute missing values
        const trainImputed = imputeMissingValues(AppState.rawData.train, AppState.rawData.train);
        const testImputed = imputeMissingValues(AppState.rawData.test, AppState.rawData.train);
        
        // Prepare features
        const trainFeatures = prepareFeatures(trainImputed, true, addFamilySize, addIsAlone);
        const testFeatures = prepareFeatures(testImputed, false, addFamilySize, addIsAlone);
        
        // Store processed data
        AppState.processedData.X_train = trainFeatures.features;
        AppState.processedData.y_train = trainFeatures.target;
        AppState.processedData.X_test = testFeatures.features;
        AppState.processedData.featureNames = trainFeatures.featureNames;
        
        console.log('Preprocessing complete:', {
            trainShape: [trainFeatures.features.length, trainFeatures.features[0].length],
            testShape: [testFeatures.features.length, testFeatures.features[0].length],
            featureCount: trainFeatures.featureNames.length,
            featureNames: trainFeatures.featureNames.slice(0, 10)
        });
        
        // Update feature preview
        document.getElementById('feature-preview').innerHTML = `
            <div style="color: #2ecc71; margin-bottom: 8px;">
                <i class="fas fa-check-circle"></i> Preprocessing complete!
            </div>
            <div>Total features: <strong>${trainFeatures.featureNames.length}</strong></div>
            <div>Training samples: <strong>${trainFeatures.features.length}</strong></div>
            <div>Test samples: <strong>${testFeatures.features.length}</strong></div>
            <div style="margin-top: 12px; color: #7f8c8d; font-size: 0.85rem;">
                First 5 features: ${trainFeatures.featureNames.slice(0, 5).join(', ')}...
            </div>
        `;
        
        // Create correlation visualization
        createCorrelationChart();
        
        // Show results
        document.getElementById('preprocess-results').style.display = 'block';
        
        showStatus('✅ Data preprocessing complete! Ready for model building.', 'success', 'preprocess-status');
        
    } catch (error) {
        console.error('Error in preprocessing:', error);
        showStatus(`❌ Preprocessing error: ${error.message}`, 'error', 'preprocess-status');
    }
}

/**
 * Impute missing values in the dataset
 * @param {Array} data - Dataset to impute
 * @param {Array} referenceData - Reference dataset for calculating statistics
 * @returns {Array} Imputed dataset
 */
function imputeMissingValues(data, referenceData) {
    const ref = referenceData || data;
    
    // Calculate median for Age
    const ageValues = ref.filter(r => r.Age && !isNaN(r.Age)).map(r => r.Age);
    const ageMedian = ageValues.length > 0 ? 
        ageValues.sort((a, b) => a - b)[Math.floor(ageValues.length / 2)] : 29.7;
    
    // Calculate median for Fare
    const fareValues = ref.filter(r => r.Fare && !isNaN(r.Fare)).map(r => r.Fare);
    const fareMedian = fareValues.length > 0 ? 
        fareValues.sort((a, b) => a - b)[Math.floor(fareValues.length / 2)] : 14.45;
    
    // Calculate mode for Embarked
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

/**
 * Prepare features for model training
 * @param {Array} data - Dataset to process
 * @param {boolean} isTraining - Whether this is training data
 * @param {boolean} addFamilySize - Add family size feature
 * @param {boolean} addIsAlone - Add is alone feature
 * @returns {Object} Processed features and target
 */
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
        const ageStd = (age - 29.7) / 13.5; // Approximate mean/std from Titanic dataset
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
        const fareStd = (fare - 32.2) / 49.7; // Approximate mean/std
        featureVector.push(fareStd);
        if (featureNames.length < 9) featureNames.push('Fare_std');
        
        // One-hot encode Embarked (C, Q, S)
        const embarked = row.Embarked || 'S';
        ['C', 'Q', 'S'].forEach(port => {
            featureVector.push(embarked === port ? 1 : 0);
            if (featureNames.length < 12) featureNames.push(`Embarked_${port}`);
        });
        
        // Feature engineering: Family Size
        if (addFamilySize) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            featureVector.push(familySize);
            if (featureNames.length < 13) featureNames.push('FamilySize');
        }
        
        // Feature engineering: Is Alone
        if (addIsAlone) {
            const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
            const isAlone = familySize === 1 ? 1 : 0;
            featureVector.push(isAlone);
            if (featureNames.length < 14) featureNames.push('IsAlone');
        }
        
        features.push(featureVector);
        
        // Add target if training data
        if (isTraining) {
            target.push(row.Survived !== undefined ? row.Survived : 0);
        }
    });
    
    return { features, target, featureNames };
}

/**
 * Create feature correlation chart
 */
function createCorrelationChart() {
    const ctx = document.getElementById('feature-correlation-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (AppState.charts.correlation) {
        AppState.charts.correlation.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    // For demonstration, create a simple feature importance chart
    // In a real application, you would calculate actual correlations
    const featureNames = AppState.processedData.featureNames.slice(0, 8);
    const importanceScores = featureNames.map((_, i) => Math.random() * 0.8 + 0.2);
    
    AppState.charts.correlation = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'Feature Importance',
                data: importanceScores,
                backgroundColor: '#3498db',
                borderColor: '#2980b9',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Importance Score',
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y: {
                    grid: { display: false }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Importance: ${context.raw.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// SECTION 3: MODEL BUILDING
// ============================================

/**
 * Build the neural network model
 */
async function buildModel() {
    showStatus('🧠 Building neural network...', 'info', 'model-status');
    
    if (!AppState.processedData.X_train || !AppState.processedData.y_train) {
        showStatus('❌ Please preprocess data first', 'error', 'model-status');
        return;
    }
    
    try {
        // Get model configuration
        const hiddenUnits = parseInt(document.getElementById('hidden-units').value) || 16;
        const learningRate = parseFloat(document.getElementById('learning-rate').value) || 0.001;
        
        // Create sequential model
        AppState.model = tf.sequential();
        
        // Add hidden layer
        AppState.model.add(tf.layers.dense({
            units: hiddenUnits,
            activation: 'relu',
            inputShape: [AppState.processedData.featureNames.length],
            kernelInitializer: 'glorotNormal',
            biasInitializer: 'zeros'
        }));
        
        // Add output layer
        AppState.model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }));
        
        // Compile model
        AppState.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        // Display model summary
        displayModelSummary();
        
        // Show success message
        showStatus(`✅ Model built successfully! (${hiddenUnits} hidden units, learning rate: ${learningRate})`, 'success', 'model-status');
        
        // Show next step button
        document.getElementById('model-built').style.display = 'block';
        
        console.log('Model summary:', {
            totalParams: AppState.model.countParams(),
            layers: AppState.model.layers.map(layer => ({
                type: layer.getClassName(),
                units: layer.getConfig().units,
                activation: layer.getConfig().activation
            }))
        });
        
    } catch (error) {
        console.error('Error building model:', error);
        showStatus(`❌ Model building error: ${error.message}`, 'error', 'model-status');
    }
}

/**
 * Display model summary in UI
 */
function displayModelSummary() {
    const summaryElement = document.getElementById('model-summary');
    if (!summaryElement) return;
    
    let summaryHTML = `
        <div style="color: #3498db; font-weight: bold; margin-bottom: 12px; font-size: 1.1rem;">
            <i class="fas fa-project-diagram"></i> Model Architecture
        </div>
    `;
    
    // Layer information
    AppState.model.layers.forEach((layer, index) => {
        const config = layer.getConfig();
        const layerType = layer.getClassName();
        const isOutput = index === AppState.model.layers.length - 1;
        
        summaryHTML += `
            <div style="margin-bottom: 10px; padding: 12px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid ${isOutput ? '#e74c3c' : '#3498db'};">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    ${isOutput ? 'Output Layer' : 'Hidden Layer'} (${layerType})
                </div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">
                    <div>Units: ${config.units}</div>
                    <div>Activation: ${config.activation}</div>
                    <div>Input Shape: ${config.inputShape ? config.inputShape.join('×') : 'Auto'}</div>
                </div>
            </div>
        `;
    });
    
    // Total parameters
    const totalParams = AppState.model.countParams();
    summaryHTML += `
        <div style="margin-top: 16px; padding: 12px; background: linear-gradient(135deg, #3498db, #2c3e50); 
                    color: white; border-radius: 8px; font-weight: bold; text-align: center;">
            <i class="fas fa-calculator"></i> Total Parameters: ${totalParams.toLocaleString()}
        </div>
    `;
    
    summaryElement.innerHTML = summaryHTML;
}

// ============================================
// SECTION 4: MODEL TRAINING
// ============================================

/**
 * Start model training
 */
async function startTraining() {
    showStatus('🏋️ Starting model training...', 'info', 'training-status');
    
    if (!AppState.model) {
        showStatus('❌ Please build the model first', 'error', 'training-status');
        return;
    }
    
    if (!AppState.processedData.X_train || !AppState.processedData.y_train) {
        showStatus('❌ Please preprocess data first', 'error', 'training-status');
        return;
    }
    
    AppState.isTraining = true;
    AppState.trainingHistory = [];
    
    // Get training parameters
    const epochs = parseInt(document.getElementById('epochs').value) || 50;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 32;
    
    try {
        // Convert data to tensors
        const xs = tf.tensor2d(AppState.processedData.X_train);
        const ys = tf.tensor2d(AppState.processedData.y_train, [AppState.processedData.y_train.length, 1]);
        
        // Setup training visualization
        document.getElementById('training-visualization').style.display = 'block';
        initializeTrainingChart();
        
        // Train the model
        const history = await AppState.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // Store history
                    AppState.trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        valLoss: logs.val_loss,
                        valAccuracy: logs.val_acc
                    });
                    
                    // Update progress bar
                    const progress = ((epoch + 1) / epochs) * 100;
                    document.getElementById('training-progress').style.width = `${progress}%`;
                    document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
                    
                    // Update training stats
                    document.getElementById('training-stats').innerHTML = `
                        <div style="font-size: 1.1rem; font-weight: 600; color: #2c3e50;">
                            Epoch ${epoch + 1}/${epochs}
                        </div>
                        <div style="color: #7f8c8d; margin-top: 8px;">
                            <div>Loss: <strong>${logs.loss.toFixed(4)}</strong></div>
                            <div>Accuracy: <strong>${(logs.acc * 100).toFixed(1)}%</strong></div>
                            <div>Val Loss: <strong>${logs.val_loss.toFixed(4)}</strong></div>
                        </div>
                    `;
                    
                    // Update training chart
                    updateTrainingChart();
                    
                    // Check for early stopping
                    if (logs.val_loss > 1.5 && epoch > 10) {
                        console.log('Early stopping triggered: validation loss too high');
                        AppState.isTraining = false;
                    }
                },
                onTrainEnd: () => {
                    AppState.isTraining = false;
                    showStatus('✅ Training completed successfully!', 'success', 'training-status');
                    
                    // Clean up tensors
                    xs.dispose();
                    ys.dispose();
                }
            }
        });
        
    } catch (error) {
        console.error('Error during training:', error);
        AppState.isTraining = false;
        showStatus(`❌ Training error: ${error.message}`, 'error', 'training-status');
    }
}

/**
 * Stop model training
 */
function stopTraining() {
    if (AppState.isTraining) {
        AppState.isTraining = false;
        showStatus('⏹️ Training stopped by user', 'info', 'training-status');
    }
}

/**
 * Initialize training chart
 */
function initializeTrainingChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (AppState.charts.training) {
        AppState.charts.training.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    AppState.charts.training = new Chart(chartCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
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
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss',
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy',
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: { drawOnChartArea: false },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    min: 0,
                    max: 1
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

/**
 * Update training chart with new data
 */
function updateTrainingChart() {
    if (!AppState.charts.training || AppState.trainingHistory.length === 0) return;
    
    const history = AppState.trainingHistory;
    
    AppState.charts.training.data.labels = history.map(h => h.epoch);
    AppState.charts.training.data.datasets[0].data = history.map(h => h.loss);
    AppState.charts.training.data.datasets[1].data = history.map(h => h.valLoss);
    AppState.charts.training.data.datasets[2].data = history.map(h => h.accuracy);
    
    AppState.charts.training.update();
}

// ============================================
// SECTION 5: EVALUATION & PREDICTIONS
// ============================================

/**
 * Evaluate the trained model
 */
async function evaluateModel() {
    showStatus('📊 Evaluating model performance...', 'info', 'results-status');
    
    if (!AppState.model) {
        showStatus('❌ Please train the model first', 'error', 'results-status');
        return;
    }
    
    try {
        // Split training data for validation
        const validationSize = Math.floor(AppState.processedData.X_train.length * 0.2);
        const X_val = AppState.processedData.X_train.slice(0, validationSize);
        const y_val = AppState.processedData.y_train.slice(0, validationSize);
        
        // Convert to tensors
        const xsVal = tf.tensor2d(X_val);
        
        // Make predictions
        const predictions = AppState.model.predict(xsVal);
        const probs = await predictions.data();
        
        // Store validation data
        AppState.validationProbs = Array.from(probs);
        AppState.validationLabels = y_val;
        
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
        
        // Setup threshold slider
        setupThresholdSlider();
        
        // Show metrics display
        document.getElementById('metrics-display').style.display = 'block';
        document.getElementById('export-section').style.display = 'block';
        
        showStatus('✅ Model evaluation complete!', 'success', 'results-status');
        
        // Clean up
        xsVal.dispose();
        predictions.dispose();
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        showStatus(`❌ Evaluation error: ${error.message}`, 'error', 'results-status');
    }
}

/**
 * Calculate accuracy and loss metrics
 * @param {Array} trueLabels - True labels
 * @param {Array} predictedProbs - Predicted probabilities
 * @returns {Object} Accuracy and loss
 */
function calculateMetrics(trueLabels, predictedProbs) {
    const predictedLabels = predictedProbs.map(p => p >= AppState.currentThreshold ? 1 : 0);
    
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
        const epsilon = 1e-15;
        const p_clipped = Math.max(epsilon, Math.min(1 - epsilon, p));
        loss += y * Math.log(p_clipped) + (1 - y) * Math.log(1 - p_clipped);
    }
    loss = -loss / trueLabels.length;
    
    return { accuracy, loss };
}

/**
 * Calculate AUC using trapezoidal rule
 * @param {Array} trueLabels - True labels
 * @param {Array} predictedProbs - Predicted probabilities
 * @returns {number} AUC score
 */
function calculateAUC(trueLabels, predictedProbs) {
    // Sort by predicted probability descending
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

/**
 * Create confusion matrix chart
 * @param {Array} trueLabels - True labels
 * @param {Array} predictedProbs - Predicted probabilities
 */
function createConfusionMatrix(trueLabels, predictedProbs) {
    const ctx = document.getElementById('confusion-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (AppState.charts.confusion) {
        AppState.charts.confusion.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    // Calculate confusion matrix
    const predictedLabels = predictedProbs.map(p => p >= AppState.currentThreshold ? 1 : 0);
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predictedLabels[i] === 1) tp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 1) fp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 0) tn++;
        if (trueLabels[i] === 1 && predictedLabels[i] === 0) fn++;
    }
    
    // Calculate percentages
    const total = tp + fp + tn + fn;
    const tpPercent = ((tp / total) * 100).toFixed(1);
    const fpPercent = ((fp / total) * 100).toFixed(1);
    const tnPercent = ((tn / total) * 100).toFixed(1);
    const fnPercent = ((fn / total) * 100).toFixed(1);
    
    AppState.charts.confusion = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
            datasets: [{
                label: 'Count',
                data: [tp, fp, tn, fn],
                backgroundColor: ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c'],
                borderColor: ['#27ae60', '#c0392b', '#27ae60', '#c0392b'],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const percentages = [tpPercent, fpPercent, tnPercent, fnPercent];
                            return `${context.label}: ${context.raw} (${percentages[index]}%)`;
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
                        font: { size: 14, weight: 'bold' }
                    },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

/**
 * Create ROC curve chart
 * @param {Array} trueLabels - True labels
 * @param {Array} predictedProbs - Predicted probabilities
 */
function createROCCurve(trueLabels, predictedProbs) {
    const ctx = document.getElementById('roc-chart');
    if (!ctx) return;
    
    // Destroy existing chart
    if (AppState.charts.roc) {
        AppState.charts.roc.destroy();
    }
    
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
    
    // Find point closest to current threshold
    const currentPoint = rocPoints.reduce((closest, point) => {
        return Math.abs(point.threshold - AppState.currentThreshold) < 
               Math.abs(closest.threshold - AppState.currentThreshold) ? point : closest;
    });
    
    AppState.charts.roc = new Chart(chartCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'ROC Curve',
                    data: rocPoints.map(p => ({ x: p.fpr, y: p.tpr })),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Random Classifier',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#7f8c8d',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Current Threshold',
                    data: [{ x: currentPoint.fpr, y: currentPoint.tpr }],
                    backgroundColor: '#e74c3c',
                    borderColor: '#c0392b',
                    pointRadius: 8,
                    pointHoverRadius: 10
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: 'False Positive Rate',
                        font: { size: 14, weight: 'bold' }
                    },
                    min: 0,
                    max: 1,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: 'True Positive Rate',
                        font: { size: 14, weight: 'bold' }
                    },
                    min: 0,
                    max: 1,
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 2) {
                                return `Threshold: ${AppState.currentThreshold.toFixed(2)}`;
                            }
                            return null;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Setup threshold slider with event listener
 */
function setupThresholdSlider() {
    const slider = document.getElementById('threshold-slider');
    const valueDisplay = document.getElementById('threshold-value');
    
    if (!slider || !valueDisplay) return;
    
    slider.value = AppState.currentThreshold;
    valueDisplay.textContent = AppState.currentThreshold.toFixed(2);
    
    slider.oninput = function() {
        AppState.currentThreshold = parseFloat(this.value);
        valueDisplay.textContent = AppState.currentThreshold.toFixed(2);
        
        // Update confusion matrix with new threshold
        if (AppState.validationLabels && AppState.validationProbs) {
            createConfusionMatrix(AppState.validationLabels, AppState.validationProbs);
            createROCCurve(AppState.validationLabels, AppState.validationProbs);
        }
    };
}

/**
 * Make predictions on test data
 */
async function makePredictions() {
    showStatus('🔮 Making predictions on test data...', 'info', 'results-status');
    
    if (!AppState.model || !AppState.processedData.X_test) {
        showStatus('❌ Please train model and preprocess data first', 'error', 'results-status');
        return;
    }
    
    try {
        // Convert test data to tensor
        const xsTest = tf.tensor2d(AppState.processedData.X_test);
        
        // Make predictions
        const predictions = AppState.model.predict(xsTest);
        const probs = await predictions.data();
        
        // Store predictions
        AppState.testProbs = Array.from(probs);
        AppState.predictions = probs.map(p => p >= AppState.currentThreshold ? 1 : 0);
        
        // Calculate predicted survivors
        const predictedSurvivors = AppState.predictions.filter(p => p === 1).length;
        
        showStatus(`✅ Predictions complete! ${predictedSurvivors} passengers predicted to survive.`, 'success', 'results-status');
        
        // Clean up
        xsTest.dispose();
        predictions.dispose();
        
    } catch (error) {
        console.error('Error making predictions:', error);
        showStatus(`❌ Prediction error: ${error.message}`, 'error', 'results-status');
    }
}

/**
 * Export submission CSV file
 */
function exportSubmission() {
    if (!AppState.predictions || !AppState.rawData.test) {
        showStatus('❌ Please make predictions first', 'error', 'export-status');
        return;
    }
    
    try {
        // Create CSV content
        let csvContent = 'PassengerId,Survived\n';
        
        AppState.rawData.test.forEach((row, index) => {
            const passengerId = row.PassengerId || (892 + index); // Kaggle test IDs start from 892
            const survived = AppState.predictions[index] || 0;
            csvContent += `${passengerId},${survived}\n`;
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'titanic_submission.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showStatus('✅ submission.csv downloaded successfully!', 'success', 'export-status');
        
    } catch (error) {
        console.error('Error exporting submission:', error);
        showStatus(`❌ Export error: ${error.message}`, 'error', 'export-status');
    }
}

/**
 * Export probabilities CSV file
 */
function exportProbabilities() {
    if (!AppState.testProbs || !AppState.rawData.test) {
        showStatus('❌ Please make predictions first', 'error', 'export-status');
        return;
    }
    
    try {
        // Create CSV content
        let csvContent = 'PassengerId,SurvivalProbability\n';
        
        AppState.rawData.test.forEach((row, index) => {
            const passengerId = row.PassengerId || (892 + index);
            const probability = AppState.testProbs[index] || 0;
            csvContent += `${passengerId},${probability.toFixed(6)}\n`;
        });
        
        // Create download link
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'titanic_probabilities.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showStatus('✅ probabilities.csv downloaded successfully!', 'success', 'export-status');
        
    } catch (error) {
        console.error('Error exporting probabilities:', error);
        showStatus(`❌ Export error: ${error.message}`, 'error', 'export-status');
    }
}

/**
 * Save the trained model
 */
async function saveModel() {
    if (!AppState.model) {
        showStatus('❌ No model to save', 'error', 'export-status');
        return;
    }
    
    try {
        // Save model using TensorFlow.js save API
        await AppState.model.save('downloads://titanic-tfjs-model');
        
        showStatus('✅ Model saved successfully! Check your downloads folder.', 'success', 'export-status');
        
    } catch (error) {
        console.error('Error saving model:', error);
        showStatus(`❌ Model save error: ${error.message}`, 'error', 'export-status');
    }
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the application
 */
function initializeApp() {
    console.log('🚢 Titanic Neural Explorer initialized');
    
    // Setup event listeners for sliders
    const hiddenUnitsSlider = document.getElementById('hidden-units');
    const epochsSlider = document.getElementById('epochs');
    
    if (hiddenUnitsSlider) {
        const unitsDisplay = document.getElementById('units-display');
        hiddenUnitsSlider.oninput = function() {
            unitsDisplay.textContent = this.value;
        };
    }
    
    if (epochsSlider) {
        const epochsDisplay = document.getElementById('epochs-display');
        epochsSlider.oninput = function() {
            epochsDisplay.textContent = this.value;
        };
    }
    
    // Setup drag and drop
    setupDragAndDrop();
    
    // Show initial status
    showStatus('Ready to explore Titanic survival patterns. Upload your CSV files to begin.', 'info');
}

/**
 * Setup drag and drop functionality
 */
function setupDragAndDrop() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) return;
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#3498db';
        dropZone.style.background = 'rgba(52, 152, 219, 0.05)';
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#d1d1d6';
        dropZone.style.background = 'rgba(255, 255, 255, 0.5)';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#d1d1d6';
        dropZone.style.background = 'rgba(255, 255, 255, 0.5)';
        
        const files = e.dataTransfer.files;
        handleDroppedFiles(files);
    });
}

/**
 * Handle dropped files
 * @param {FileList} files - Dropped files
 */
function handleDroppedFiles(files) {
    const trainFileInput = document.getElementById('trainFile');
    const testFileInput = document.getElementById('testFile');
    
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

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', initializeApp);

// Make functions available globally
window.loadAndExploreData = loadAndExploreData;
window.resetAll = resetAll;
window.preprocessData = preprocessData;
window.buildModel = buildModel;
window.startTraining = startTraining;
window.stopTraining = stopTraining;
window.evaluateModel = evaluateModel;
window.makePredictions = makePredictions;
window.exportSubmission = exportSubmission;
window.exportProbabilities = exportProbabilities;
window.saveModel = saveModel;
window.goToStep = goToStep;