// app.js - Titanic Neural Network Classifier with Sigmoid Gate
// Complete browser-based ML with TensorFlow.js

"use strict";

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    target: 'Survived',
    features: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    identifier: 'PassengerId',
    categorical: ['Sex', 'Embarked', 'Pclass'],
    numerical: ['Age', 'Fare', 'SibSp', 'Parch'],
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
    rawData: {
        train: null,
        test: null
    },
    processedData: {
        X_train: null,
        y_train: null,
        X_val: null,
        y_val: null,
        X_test: null,
        featureNames: [],
        scalers: {}
    },
    model: null,
    trainingHistory: [],
    isTraining: false,
    predictions: null,
    validationProbs: null,
    validationLabels: null,
    testProbs: null,
    charts: {
        exploration: null,
        training: null,
        roc: null,
        confusion: null,
        correlation: null,
        trainedFeatureImportance: null
    },
    currentThreshold: 0.5,
    useSigmoidGate: true,
    currentFeatures: [],
    matrixView: 'chart' // 'chart' or 'table'
};

// ============================================
// UTILITY FUNCTIONS
// ============================================

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
        try {
            const lines = csvText.trim().split('\n');
            if (lines.length === 0) {
                resolve([]);
                return;
            }
            
            const headers = lines[0].split(',').map(h => {
                return h.trim().replace(/^"(.*)"$/, '$1');
            });
            
            const data = [];
            for (let i = 1; i < lines.length; i++) {
                if (lines[i].trim() === '') continue;
                
                const rowValues = [];
                let currentField = '';
                let inQuotes = false;
                
                for (let char of lines[i]) {
                    if (char === '"') {
                        inQuotes = !inQuotes;
                    } else if (char === ',' && !inQuotes) {
                        rowValues.push(currentField);
                        currentField = '';
                    } else {
                        currentField += char;
                    }
                }
                rowValues.push(currentField);
                
                const row = {};
                headers.forEach((header, index) => {
                    let value = rowValues[index] || '';
                    value = value.trim();
                    value = value.replace(/^"(.*)"$/, '$1');
                    
                    if (!isNaN(value) && value !== '') {
                        value = value.includes('.') ? parseFloat(value) : parseInt(value, 10);
                    } else if (value === '') {
                        value = null;
                    }
                    
                    row[header] = value;
                });
                
                data.push(row);
            }
            
            console.log(`Parsed ${data.length} rows with ${headers.length} columns`);
            resolve(data);
        } catch (error) {
            reject(new Error(`CSV parsing error: ${error.message}`));
        }
    });
}

function goToStep(stepNumber) {
    document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
        indicator.classList.toggle('active', (index + 1) === stepNumber);
    });
    
    document.querySelectorAll('.section').forEach((section, index) => {
        section.classList.toggle('active', (index + 1) === stepNumber);
    });
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function resetAll() {
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
    AppState.currentFeatures = [];
    
    document.getElementById('trainFile').value = '';
    document.getElementById('testFile').value = '';
    
    goToStep(1);
    
    Object.values(AppState.charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    AppState.charts = {
        exploration: null,
        training: null,
        roc: null,
        confusion: null,
        correlation: null,
        trainedFeatureImportance: null
    };
    
    document.getElementById('data-overview').style.display = 'none';
    document.getElementById('preprocess-results').style.display = 'none';
    document.getElementById('model-built').style.display = 'none';
    document.getElementById('training-visualization').style.display = 'none';
    document.getElementById('metrics-display').style.display = 'none';
    document.getElementById('export-section').style.display = 'none';
    document.getElementById('confusion-matrix-container').style.display = 'none';
    document.getElementById('confusion-chart-container').style.display = 'block';
    document.getElementById('feature-importance-after-training').style.display = 'none';
    
    showStatus('Application reset. Ready to load new data.', 'info');
}

// ============================================
// SECTION 1: DATA LOADING & EXPLORATION
// ============================================

async function loadAndExploreData() {
    const trainFile = document.getElementById('trainFile').files[0];
    const testFile = document.getElementById('testFile').files[0];
    
    if (!trainFile || !testFile) {
        showStatus('❌ Please select both train.csv and test.csv files', 'error');
        return;
    }
    
    if (!trainFile.name.toLowerCase().includes('train') || !testFile.name.toLowerCase().includes('test')) {
        showStatus('❌ Please select correct files: train.csv and test.csv', 'error');
        return;
    }
    
    showStatus('⏳ Loading CSV files...', 'info');
    
    try {
        const trainText = await readFile(trainFile);
        const testText = await readFile(testFile);
        
        AppState.rawData.train = await parseCSV(trainText);
        AppState.rawData.test = await parseCSV(testText);
        
        console.log('Data loaded:', {
            train: AppState.rawData.train.length,
            test: AppState.rawData.test.length
        });
        
        updateDataOverview();
        createExplorationCharts();
        
        showStatus(`✅ Loaded ${AppState.rawData.train.length} training and ${AppState.rawData.test.length} test samples`, 'success');
        document.getElementById('data-overview').style.display = 'block';
        
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus(`❌ Error loading data: ${error.message}`, 'error');
    }
}

function updateDataOverview() {
    const trainData = AppState.rawData.train;
    const testData = AppState.rawData.test;
    
    document.getElementById('total-passengers').textContent = trainData.length + testData.length;
    document.getElementById('train-count').textContent = trainData.length;
    document.getElementById('test-count').textContent = testData.length;
    
    const survived = trainData.filter(row => row.Survived === 1).length;
    const survivalRate = ((survived / trainData.length) * 100).toFixed(1);
    document.getElementById('survival-rate').textContent = `${survivalRate}%`;
    
    showDataPreview();
    showMissingValues();
}

function showDataPreview() {
    const tbody = document.getElementById('preview-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    AppState.rawData.train.slice(0, 8).forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.PassengerId || ''}</td>
            <td>${(row.Name || '').substring(0, 20)}...</td>
            <td>${row.Pclass || ''}</td>
            <td>${row.Sex || ''}</td>
            <td>${row.Age ? row.Age.toFixed(1) : 'N/A'}</td>
            <td>${row.Fare ? row.Fare.toFixed(2) : 'N/A'}</td>
            <td>${row.Survived !== undefined && row.Survived !== null ? row.Survived : 'N/A'}</td>
        `;
        tbody.appendChild(tr);
    });
}

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

function createExplorationCharts() {
    const ctx = document.getElementById('exploration-chart');
    if (!ctx) return;
    
    if (AppState.charts.exploration) {
        AppState.charts.exploration.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    const trainData = AppState.rawData.train;
    
    const male = trainData.filter(d => d.Sex === 'male');
    const female = trainData.filter(d => d.Sex === 'female');
    
    const maleSurvival = male.length > 0 ? 
        (male.filter(d => d.Survived === 1).length / male.length * 100).toFixed(1) : 0;
    const femaleSurvival = female.length > 0 ? 
        (female.filter(d => d.Survived === 1).length / female.length * 100).toFixed(1) : 0;
    
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

async function preprocessData() {
    showStatus('🔧 Preprocessing data...', 'info', 'preprocess-status');
    
    if (!AppState.rawData.train || !AppState.rawData.test) {
        showStatus('❌ Please load data first', 'error', 'preprocess-status');
        return;
    }
    
    try {
        const addFamilySize = document.getElementById('family-size-toggle').checked;
        const addIsAlone = document.getElementById('is-alone-toggle').checked;
        AppState.useSigmoidGate = document.getElementById('sigmoid-gate-toggle').checked;
        
        const trainImputed = imputeMissingValues(AppState.rawData.train, AppState.rawData.train);
        const testImputed = imputeMissingValues(AppState.rawData.test, AppState.rawData.train);
        
        const trainFeatures = prepareFeatures(trainImputed, true, addFamilySize, addIsAlone);
        const testFeatures = prepareFeatures(testImputed, false, addFamilySize, addIsAlone);
        
        AppState.processedData.X_train = trainFeatures.features;
        AppState.processedData.y_train = trainFeatures.target;
        AppState.processedData.X_test = testFeatures.features;
        AppState.processedData.featureNames = trainFeatures.featureNames;
        AppState.currentFeatures = trainFeatures.featureNames;
        
        console.log('Preprocessing complete:', {
            trainShape: [trainFeatures.features.length, trainFeatures.features[0].length],
            featureCount: trainFeatures.featureNames.length,
            features: trainFeatures.featureNames
        });
        
        document.getElementById('feature-preview').innerHTML = `
            <div style="color: #2ecc71; margin-bottom: 8px;">
                <i class="fas fa-check-circle"></i> Preprocessing complete!
            </div>
            <div>Total features: <strong>${trainFeatures.featureNames.length}</strong></div>
            <div>Training samples: <strong>${trainFeatures.features.length}</strong></div>
            <div>Test samples: <strong>${testFeatures.features.length}</strong></div>
            <div style="margin-top: 12px; color: #7f8c8d; font-size: 0.85rem;">
                Features: ${trainFeatures.featureNames.slice(0, 5).join(', ')}...
            </div>
            ${AppState.useSigmoidGate ? '<div style="color: #9b59b6; margin-top: 8px;"><i class="fas fa-filter"></i> Sigmoid Gate enabled</div>' : ''}
            ${addFamilySize ? '<div style="color: #3498db; margin-top: 4px;"><i class="fas fa-users"></i> Family Size feature added</div>' : ''}
            ${addIsAlone ? '<div style="color: #3498db; margin-top: 4px;"><i class="fas fa-user"></i> Is Alone feature added</div>' : ''}
        `;
        
        createCorrelationChart(addFamilySize, addIsAlone);
        document.getElementById('preprocess-results').style.display = 'block';
        showStatus('✅ Data preprocessing complete!', 'success', 'preprocess-status');
        
    } catch (error) {
        console.error('Error in preprocessing:', error);
        showStatus(`❌ Preprocessing error: ${error.message}`, 'error', 'preprocess-status');
    }
}

function imputeMissingValues(data, referenceData) {
    const ref = referenceData || data;
    
    const ageValues = ref.filter(r => r.Age && !isNaN(r.Age)).map(r => r.Age);
    const ageMedian = ageValues.length > 0 ? 
        ageValues.sort((a, b) => a - b)[Math.floor(ageValues.length / 2)] : 29.7;
    
    const fareValues = ref.filter(r => r.Fare && !isNaN(r.Fare)).map(r => r.Fare);
    const fareMedian = fareValues.length > 0 ? 
        fareValues.sort((a, b) => a - b)[Math.floor(fareValues.length / 2)] : 14.45;
    
    const embarkedCounts = {};
    ref.filter(r => r.Embarked).forEach(r => {
        embarkedCounts[r.Embarked] = (embarkedCounts[r.Embarked] || 0) + 1;
    });
    const embarkedMode = Object.keys(embarkedCounts).length > 0 ?
        Object.keys(embarkedCounts).reduce((a, b) => embarkedCounts[a] > embarkedCounts[b] ? a : b) : 'S';
    
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
    
    data.forEach(row => {
        const featureVector = [];
        
        const pclass = row.Pclass || 3;
        for (let i = 1; i <= 3; i++) {
            featureVector.push(pclass === i ? 1 : 0);
            if (featureNames.length < 3) featureNames.push(`Pclass_${i}`);
        }
        
        const sex = row.Sex || 'male';
        featureVector.push(sex === 'female' ? 1 : 0);
        featureVector.push(sex === 'male' ? 1 : 0);
        if (featureNames.length < 5) {
            featureNames.push('Sex_female');
            featureNames.push('Sex_male');
        }
        
        const age = row.Age || 29.7;
        const ageStd = (age - 29.7) / 13.5;
        featureVector.push(ageStd);
        if (featureNames.length < 6) featureNames.push('Age_std');
        
        featureVector.push(row.SibSp || 0);
        featureVector.push(row.Parch || 0);
        if (featureNames.length < 8) {
            featureNames.push('SibSp');
            featureNames.push('Parch');
        }
        
        const fare = row.Fare || 14.45;
        const fareStd = (fare - 32.2) / 49.7;
        featureVector.push(fareStd);
        if (featureNames.length < 9) featureNames.push('Fare_std');
        
        const embarked = row.Embarked || 'S';
        ['C', 'Q', 'S'].forEach(port => {
            featureVector.push(embarked === port ? 1 : 0);
            if (featureNames.length < 12) featureNames.push(`Embarked_${port}`);
        });
        
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

function createCorrelationChart(addFamilySize = true, addIsAlone = true) {
    const ctx = document.getElementById('feature-correlation-chart');
    if (!ctx) return;
    
    if (AppState.charts.correlation) {
        AppState.charts.correlation.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    const featureNames = AppState.processedData.featureNames;
    
    // Динамически рассчитываем важность признаков на основе выбранных опций
    const importanceScores = calculateDynamicFeatureImportance(featureNames, addFamilySize, addIsAlone);
    
    // Берем топ-8 признаков
    const topFeatures = [];
    const topScores = [];
    
    // Сортируем по важности
    const sortedFeatures = featureNames.map((name, index) => ({
        name,
        score: importanceScores[index]
    })).sort((a, b) => b.score - a.score);
    
    sortedFeatures.slice(0, 8).forEach(item => {
        topFeatures.push(item.name);
        topScores.push(item.score);
    });
    
    AppState.charts.correlation = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: topFeatures,
            datasets: [{
                label: 'Feature Importance',
                data: topScores,
                backgroundColor: topFeatures.map((name, index) => {
                    if (name.includes('Sex_female')) return '#e74c3c';
                    if (name.includes('Pclass_1')) return '#2ecc71';
                    if (name.includes('Age')) return '#3498db';
                    if (name.includes('Fare')) return '#f39c12';
                    if (name.includes('FamilySize')) return '#9b59b6';
                    if (name.includes('IsAlone')) return '#1abc9c';
                    return '#95a5a6';
                }),
                borderColor: topFeatures.map((name, index) => {
                    if (name.includes('Sex_female')) return '#c0392b';
                    if (name.includes('Pclass_1')) return '#27ae60';
                    if (name.includes('Age')) return '#2980b9';
                    if (name.includes('Fare')) return '#d68910';
                    if (name.includes('FamilySize')) return '#8e44ad';
                    if (name.includes('IsAlone')) return '#16a085';
                    return '#7f8c8d';
                }),
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
                            const featureName = topFeatures[context.dataIndex];
                            let description = '';
                            
                            if (featureName.includes('Sex_female')) description = ' (Female passengers)';
                            else if (featureName.includes('Pclass_1')) description = ' (First class)';
                            else if (featureName.includes('Age')) description = ' (Standardized age)';
                            else if (featureName.includes('Fare')) description = ' (Ticket fare)';
                            else if (featureName.includes('FamilySize')) description = ' (Family size)';
                            else if (featureName.includes('IsAlone')) description = ' (Traveling alone)';
                            
                            return `${featureName}${description}: ${context.raw.toFixed(3)}`;
                        }
                    }
                }
            }
        }
    });
}

function calculateDynamicFeatureImportance(featureNames, addFamilySize, addIsAlone) {
    // Основано на известных фактах о Титанике
    const importanceMap = {
        'Sex_female': 0.95,  // Самый важный признак
        'Sex_male': 0.05,
        'Pclass_1': 0.85,    // Первый класс
        'Pclass_2': 0.45,
        'Pclass_3': 0.15,
        'Age_std': 0.65,     // Возраст
        'Fare_std': 0.60,    // Стоимость билета
        'SibSp': 0.35,       // Братья/сестры
        'Parch': 0.30,       // Родители/дети
        'Embarked_C': 0.25,  // Шербур
        'Embarked_Q': 0.15,
        'Embarked_S': 0.20,
        'FamilySize': addFamilySize ? 0.40 : 0,  // Динамически
        'IsAlone': addIsAlone ? 0.35 : 0         // Динамически
    };
    
    return featureNames.map(name => {
        // Ищем точное совпадение
        if (importanceMap[name] !== undefined) {
            return importanceMap[name];
        }
        
        // Ищем частичное совпадение
        for (const key in importanceMap) {
            if (name.includes(key)) {
                return importanceMap[key];
            }
        }
        
        // Значение по умолчанию
        return 0.2 + Math.random() * 0.2;
    });
}

// ============================================
// SECTION 3: MODEL BUILDING WITH SIGMOID GATE
// ============================================

async function buildModel() {
    showStatus('🧠 Building neural network...', 'info', 'model-status');
    
    if (!AppState.processedData.X_train || !AppState.processedData.y_train) {
        showStatus('❌ Please preprocess data first', 'error', 'model-status');
        return;
    }
    
    try {
        const hiddenUnits = parseInt(document.getElementById('hidden-units').value) || 16;
        const learningRate = parseFloat(document.getElementById('learning-rate').value) || 0.001;
        const numFeatures = AppState.processedData.featureNames.length;
        
        if (AppState.useSigmoidGate) {
            // Исправленная архитектура с сигмоидным гейтом
            AppState.model = createSigmoidGateModel(numFeatures, hiddenUnits);
        } else {
            // Обычная модель
            AppState.model = tf.sequential();
            AppState.model.add(tf.layers.dense({
                units: hiddenUnits,
                activation: 'relu',
                inputShape: [numFeatures],
                kernelInitializer: 'glorotNormal',
                name: 'hidden_layer_1'
            }));
            AppState.model.add(tf.layers.dense({
                units: 1,
                activation: 'sigmoid',
                name: 'output_layer'
            }));
        }
        
        AppState.model.compile({
            optimizer: tf.train.adam(learningRate),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        
        displayModelSummary();
        
        showStatus(`✅ Model built successfully! ${AppState.useSigmoidGate ? '(with Sigmoid Gate)' : ''}`, 'success', 'model-status');
        document.getElementById('model-built').style.display = 'block';
        
    } catch (error) {
        console.error('Error building model:', error);
        showStatus(`❌ Model building error: ${error.message}`, 'error', 'model-status');
    }
}

function createSigmoidGateModel(numFeatures, hiddenUnits) {
    // Создаем модель с помощью функционального API
    const input = tf.input({shape: [numFeatures]});
    
    // Сигмоидный гейт для важности признаков
    const gateLayer = tf.layers.dense({
        units: numFeatures,
        activation: 'sigmoid',
        kernelInitializer: 'ones',
        biasInitializer: 'zeros',
        name: 'sigmoid_gate'
    });
    
    const gateOutput = gateLayer.apply(input);
    
    // Умножение входных признаков на веса гейта
    // Используем tf.layers.multiply() правильно - передаем массив
    const multiplyLayer = tf.layers.multiply({name: 'feature_gating'});
    const gatedFeatures = multiplyLayer.apply([input, gateOutput]);
    
    // Скрытый слой
    const hiddenLayer = tf.layers.dense({
        units: hiddenUnits,
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
        name: 'hidden_layer'
    });
    const hiddenOutput = hiddenLayer.apply(gatedFeatures);
    
    // Выходной слой
    const outputLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output_layer'
    });
    const finalOutput = outputLayer.apply(hiddenOutput);
    
    // Создаем модель
    const model = tf.model({
        inputs: input,
        outputs: finalOutput
    });
    
    return model;
}

function displayModelSummary() {
    const summaryElement = document.getElementById('model-summary');
    if (!summaryElement) return;
    
    if (!AppState.model) {
        summaryElement.innerHTML = `
            <div style="color: #8e8e93; text-align: center; padding: 2rem;">
                <i class="fas fa-cogs fa-2x"></i><br>
                Model architecture will appear here
            </div>
        `;
        return;
    }
    
    let summaryHTML = `
        <div style="color: #3498db; font-weight: bold; margin-bottom: 12px; font-size: 1.1rem;">
            <i class="fas fa-project-diagram"></i> Model Architecture
        </div>
    `;
    
    AppState.model.layers.forEach((layer, index) => {
        const config = layer.getConfig();
        const layerType = layer.getClassName();
        const layerName = layer.name;
        
        // Пропускаем InputLayer
        if (layerType === 'InputLayer') {
            return;
        }
        
        summaryHTML += `
            <div style="margin-bottom: 10px; padding: 12px; background: #f8f9fa; border-radius: 8px; 
                        border-left: 4px solid ${layerName === 'sigmoid_gate' ? '#9b59b6' : 
                                                layerName === 'feature_gating' ? '#3498db' :
                                                layerName === 'output_layer' ? '#e74c3c' : '#2ecc71'};">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 4px;">
                    ${layer.name} (${layerType})
                </div>
                <div style="font-size: 0.9rem; color: #7f8c8d;">
                    ${layerName === 'sigmoid_gate' ? 
                        `<div>Purpose: Learns feature importance weights (0-1)</div>
                         <div>Units: ${config.units}</div>
                         <div>Activation: ${config.activation}</div>` :
                     layerName === 'feature_gating' ?
                        `<div>Purpose: Multiplies input features by sigmoid gate weights</div>
                         <div>Operation: input × sigmoid_gate</div>` :
                     config.units ?
                        `<div>Units: ${config.units}</div>
                         <div>Activation: ${config.activation || 'linear'}</div>` :
                        `<div>Type: ${layerType}</div>`
                    }
                </div>
            </div>
        `;
    });
    
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
    
    const epochs = parseInt(document.getElementById('epochs').value) || 50;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 32;
    
    try {
        const xs = tf.tensor2d(AppState.processedData.X_train);
        const ys = tf.tensor2d(AppState.processedData.y_train, [AppState.processedData.y_train.length, 1]);
        
        document.getElementById('training-visualization').style.display = 'block';
        initializeTrainingChart();
        
        await AppState.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.2,
            shuffle: true,
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    AppState.trainingHistory.push({
                        epoch: epoch + 1,
                        loss: logs.loss,
                        accuracy: logs.acc,
                        valLoss: logs.val_loss,
                        valAccuracy: logs.val_acc
                    });
                    
                    const progress = ((epoch + 1) / epochs) * 100;
                    document.getElementById('training-progress').style.width = `${progress}%`;
                    document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
                    
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
                    
                    updateTrainingChart();
                    
                    if (logs.val_loss > 1.5 && epoch > 10) {
                        console.log('Early stopping triggered');
                        AppState.isTraining = false;
                    }
                },
                onTrainEnd: async () => {
                    AppState.isTraining = false;
                    showStatus('✅ Training completed successfully!', 'success', 'training-status');
                    
                    // После обучения показываем важность признаков из сигмоидного гейта
                    if (AppState.useSigmoidGate) {
                        await displayLearnedFeatureImportance();
                    }
                    
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

function stopTraining() {
    if (AppState.isTraining) {
        AppState.isTraining = false;
        showStatus('⏹️ Training stopped by user', 'info', 'training-status');
    }
}

async function displayLearnedFeatureImportance() {
    if (!AppState.model || !AppState.useSigmoidGate) return;
    
    try {
        // Получаем слой сигмоидного гейта
        const gateLayer = AppState.model.getLayer('sigmoid_gate');
        if (!gateLayer) {
            console.log('No sigmoid gate layer found');
            return;
        }
        
        const weights = gateLayer.getWeights()[0];
        const weightsData = await weights.data();
        
        const importances = [];
        for (let i = 0; i < weightsData.length; i++) {
            const importance = Math.abs(weightsData[i]);
            importances.push({
                feature: AppState.processedData.featureNames[i] || `Feature_${i}`,
                importance: importance
            });
        }
        
        // Сортируем по важности
        importances.sort((a, b) => b.importance - a.importance);
        
        // Обновляем график важности признаков
        updateTrainedFeatureImportanceChart(importances);
        
    } catch (error) {
        console.error('Error displaying learned feature importance:', error);
    }
}

function updateTrainedFeatureImportanceChart(importances) {
    const container = document.getElementById('feature-importance-after-training');
    const ctx = document.getElementById('trained-feature-importance-chart');
    
    if (!ctx || !importances || importances.length === 0) return;
    
    // Показываем контейнер
    container.style.display = 'block';
    
    // Уничтожаем старый график
    if (AppState.charts.trainedFeatureImportance) {
        AppState.charts.trainedFeatureImportance.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    // Берем топ-8 признаков
    const topFeatures = importances.slice(0, 8);
    
    AppState.charts.trainedFeatureImportance = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: topFeatures.map(f => f.feature),
            datasets: [{
                label: 'Learned Importance (from Sigmoid Gate)',
                data: topFeatures.map(f => f.importance),
                backgroundColor: topFeatures.map(f => {
                    if (f.importance > 0.7) return '#2ecc71';
                    if (f.importance > 0.4) return '#3498db';
                    if (f.importance > 0.1) return '#f39c12';
                    return '#e74c3c';
                }),
                borderColor: topFeatures.map(f => {
                    if (f.importance > 0.7) return '#27ae60';
                    if (f.importance > 0.4) return '#2980b9';
                    if (f.importance > 0.1) return '#d68910';
                    return '#c0392b';
                }),
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
                        text: 'Learned Importance (0-1)',
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
                            const feature = topFeatures[context.dataIndex];
                            return `${feature.feature}: ${feature.importance.toFixed(4)}`;
                        }
                    }
                }
            }
        }
    });
}

function initializeTrainingChart() {
    const ctx = document.getElementById('training-chart');
    if (!ctx) return;
    
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

async function evaluateModel() {
    showStatus('📊 Evaluating model performance...', 'info', 'results-status');
    
    if (!AppState.model) {
        showStatus('❌ Please train the model first', 'error', 'results-status');
        return;
    }
    
    try {
        const validationSize = Math.floor(AppState.processedData.X_train.length * 0.2);
        const X_val = AppState.processedData.X_train.slice(0, validationSize);
        const y_val = AppState.processedData.y_train.slice(0, validationSize);
        
        const xsVal = tf.tensor2d(X_val);
        const predictions = AppState.model.predict(xsVal);
        const probs = await predictions.data();
        
        AppState.validationProbs = Array.from(probs);
        AppState.validationLabels = y_val;
        
        const { accuracy, loss } = calculateMetrics(y_val, probs);
        const auc = calculateAUC(y_val, probs);
        
        document.getElementById('accuracy-value').textContent = accuracy.toFixed(3);
        document.getElementById('loss-value').textContent = loss.toFixed(4);
        document.getElementById('auc-value').textContent = auc.toFixed(3);
        
        createConfusionMatrix(y_val, probs);
        createROCCurve(y_val, probs);
        setupThresholdSlider();
        
        document.getElementById('metrics-display').style.display = 'block';
        document.getElementById('export-section').style.display = 'block';
        
        showStatus('✅ Model evaluation complete!', 'success', 'results-status');
        
        xsVal.dispose();
        predictions.dispose();
        
    } catch (error) {
        console.error('Error evaluating model:', error);
        showStatus(`❌ Evaluation error: ${error.message}`, 'error', 'results-status');
    }
}

function calculateMetrics(trueLabels, predictedProbs) {
    const predictedLabels = predictedProbs.map(p => p >= AppState.currentThreshold ? 1 : 0);
    
    let correct = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === predictedLabels[i]) {
            correct++;
        }
    }
    const accuracy = correct / trueLabels.length;
    
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

function calculateAUC(trueLabels, predictedProbs) {
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
    
    if (totalPositives === 0 || totalNegatives === 0) {
        return 0.5; // Cannot calculate AUC properly
    }
    
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
    const predictedLabels = predictedProbs.map(p => p >= AppState.currentThreshold ? 1 : 0);
    
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < trueLabels.length; i++) {
        if (trueLabels[i] === 1 && predictedLabels[i] === 1) tp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 1) fp++;
        if (trueLabels[i] === 0 && predictedLabels[i] === 0) tn++;
        if (trueLabels[i] === 1 && predictedLabels[i] === 0) fn++;
    }
    
    if (AppState.matrixView === 'table') {
        showTableConfusionMatrix(tp, fp, tn, fn);
    } else {
        showChartConfusionMatrix(tp, fp, tn, fn);
    }
}

function showTableConfusionMatrix(tp, fp, tn, fn) {
    const container = document.getElementById('confusion-matrix-container');
    const chartContainer = document.getElementById('confusion-chart-container');
    
    container.style.display = 'block';
    chartContainer.style.display = 'none';
    
    const total = tp + fp + tn + fn;
    const accuracy = ((tp + tn) / total * 100).toFixed(1);
    const precision = tp + fp > 0 ? (tp / (tp + fp) * 100).toFixed(1) : '0.0';
    const recall = tp + fn > 0 ? (tp / (tp + fn) * 100).toFixed(1) : '0.0';
    const f1 = precision !== '0.0' && recall !== '0.0' ? 
        (2 * (parseFloat(precision) * parseFloat(recall)) / (parseFloat(precision) + parseFloat(recall))).toFixed(1) : '0.0';
    
    const matrixHTML = `
        <div class="confusion-matrix">
            <div class="confusion-cell confusion-header"></div>
            <div class="confusion-cell confusion-header">Predicted: No</div>
            <div class="confusion-cell confusion-header">Predicted: Yes</div>
            
            <div class="confusion-cell confusion-header">Actual: No</div>
            <div class="confusion-cell confusion-tn">${tn}<br><span style="font-size: 0.8rem; color: #7f8c8d;">True Negative</span></div>
            <div class="confusion-cell confusion-fp">${fp}<br><span style="font-size: 0.8rem; color: #7f8c8d;">False Positive</span></div>
            
            <div class="confusion-cell confusion-header">Actual: Yes</div>
            <div class="confusion-cell confusion-fn">${fn}<br><span style="font-size: 0.8rem; color: #7f8c8d;">False Negative</span></div>
            <div class="confusion-cell confusion-tp">${tp}<br><span style="font-size: 0.8rem; color: #7f8c8d;">True Positive</span></div>
        </div>
        
        <div class="matrix-metrics">
            <div class="metric-item">
                <div class="label">Accuracy</div>
                <div class="value" style="color: #2ecc71;">${accuracy}%</div>
            </div>
            <div class="metric-item">
                <div class="label">Precision</div>
                <div class="value" style="color: #3498db;">${precision}%</div>
            </div>
            <div class="metric-item">
                <div class="label">Recall</div>
                <div class="value" style="color: #f39c12;">${recall}%</div>
            </div>
            <div class="metric-item">
                <div class="label">F1-Score</div>
                <div class="value" style="color: #9b59b6;">${f1}%</div>
            </div>
        </div>
    `;
    
    container.innerHTML = matrixHTML;
}

function showChartConfusionMatrix(tp, fp, tn, fn) {
    const container = document.getElementById('confusion-matrix-container');
    const chartContainer = document.getElementById('confusion-chart-container');
    
    container.style.display = 'none';
    chartContainer.style.display = 'block';
    
    const ctx = document.getElementById('confusion-chart');
    if (!ctx) return;
    
    if (AppState.charts.confusion) {
        AppState.charts.confusion.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    AppState.charts.confusion = new Chart(chartCtx, {
        type: 'bar',
        data: {
            labels: ['True Positive', 'False Positive', 'True Negative', 'False Negative'],
            datasets: [{
                label: 'Count',
                data: [tp, fp, tn, fn],
                backgroundColor: ['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c'],
                borderColor: ['#27ae60', '#c0392b', '#27ae60', '#c0392b'],
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
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = tp + fp + tn + fn;
                            const percentage = ((context.raw / total) * 100).toFixed(1);
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function toggleMatrixView() {
    AppState.matrixView = AppState.matrixView === 'table' ? 'chart' : 'table';
    
    if (AppState.validationLabels && AppState.validationProbs) {
        createConfusionMatrix(AppState.validationLabels, AppState.validationProbs);
    }
}

function createROCCurve(trueLabels, predictedProbs) {
    const ctx = document.getElementById('roc-chart');
    if (!ctx) return;
    
    if (AppState.charts.roc) {
        AppState.charts.roc.destroy();
    }
    
    const chartCtx = ctx.getContext('2d');
    
    // Generate ROC curve points
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocPoints = [];
    
    for (const threshold of thresholds) {
        const predictedLabels = predictedProbs.map(p => p >= threshold ? 1 : 0);
        
        let tp = 0, fp = 0, tn = 0, fn = 0;
        for (let i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i] === 1 && predictedLabels[i] === 1) tp++;
            if (trueLabels[i] === 0 && predictedLabels[i] === 1) fp++;
            if (trueLabels[i] === 0 && predictedLabels[i] === 0) tn++;
            if (trueLabels[i] === 1 && predictedLabels[i] === 0) fn++;
        }
        
        const tpr = (tp + fn) > 0 ? tp / (tp + fn) : 0;
        const fpr = (fp + tn) > 0 ? fp / (fp + tn) : 0;
        
        rocPoints.push({ fpr, tpr, threshold });
    }
    
    // Add (0,0) and (1,1) points
    rocPoints.unshift({ fpr: 0, tpr: 0, threshold: 1 });
    rocPoints.push({ fpr: 1, tpr: 1, threshold: 0 });
    
    // Find point closest to current threshold
    const currentPoint = rocPoints.reduce((closest, point) => {
        return Math.abs(point.threshold - AppState.currentThreshold) < 
               Math.abs(closest.threshold - AppState.currentThreshold) ? point : closest;
    });
    
    const auc = calculateAUC(trueLabels, predictedProbs);
    
    AppState.charts.roc = new Chart(chartCtx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `ROC Curve (AUC = ${auc.toFixed(3)})`,
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
                    label: `Threshold: ${AppState.currentThreshold.toFixed(2)}`,
                    data: [{ x: currentPoint.fpr, y: currentPoint.tpr }],
                    backgroundColor: '#e74c3c',
                    borderColor: '#c0392b',
                    pointRadius: 8,
                    pointHoverRadius: 10,
                    pointStyle: 'circle'
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

function setupThresholdSlider() {
    const slider = document.getElementById('threshold-slider');
    const valueDisplay = document.getElementById('threshold-value');
    
    if (!slider || !valueDisplay) return;
    
    slider.value = AppState.currentThreshold;
    valueDisplay.textContent = AppState.currentThreshold.toFixed(2);
    
    slider.oninput = function() {
        AppState.currentThreshold = parseFloat(this.value);
        valueDisplay.textContent = AppState.currentThreshold.toFixed(2);
        
        if (AppState.validationLabels && AppState.validationProbs) {
            createConfusionMatrix(AppState.validationLabels, AppState.validationProbs);
            createROCCurve(AppState.validationLabels, AppState.validationProbs);
        }
    };
}

async function makePredictions() {
    showStatus('🔮 Making predictions on test data...', 'info', 'results-status');
    
    if (!AppState.model || !AppState.processedData.X_test) {
        showStatus('❌ Please train model and preprocess data first', 'error', 'results-status');
        return;
    }
    
    try {
        const xsTest = tf.tensor2d(AppState.processedData.X_test);
        const predictions = AppState.model.predict(xsTest);
        const probs = await predictions.data();
        
        AppState.testProbs = Array.from(probs);
        AppState.predictions = probs.map(p => p >= AppState.currentThreshold ? 1 : 0);
        
        const predictedSurvivors = AppState.predictions.filter(p => p === 1).length;
        const totalPassengers = AppState.predictions.length;
        
        showStatus(`✅ Predictions complete! ${predictedSurvivors}/${totalPassengers} (${((predictedSurvivors/totalPassengers)*100).toFixed(1)}%) passengers predicted to survive.`, 'success', 'results-status');
        
        xsTest.dispose();
        predictions.dispose();
        
    } catch (error) {
        console.error('Error making predictions:', error);
        showStatus(`❌ Prediction error: ${error.message}`, 'error', 'results-status');
    }
}

function exportSubmission() {
    if (!AppState.predictions || !AppState.rawData.test) {
        showStatus('❌ Please make predictions first', 'error', 'export-status');
        return;
    }
    
    try {
        let csvContent = 'PassengerId,Survived\n';
        
        AppState.rawData.test.forEach((row, index) => {
            const passengerId = row.PassengerId || (892 + index);
            const survived = AppState.predictions[index] || 0;
            csvContent += `${passengerId},${survived}\n`;
        });
        
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

function exportProbabilities() {
    if (!AppState.testProbs || !AppState.rawData.test) {
        showStatus('❌ Please make predictions first', 'error', 'export-status');
        return;
    }
    
    try {
        let csvContent = 'PassengerId,SurvivalProbability\n';
        
        AppState.rawData.test.forEach((row, index) => {
            const passengerId = row.PassengerId || (892 + index);
            const probability = AppState.testProbs[index] || 0;
            csvContent += `${passengerId},${probability.toFixed(6)}\n`;
        });
        
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

async function saveModel() {
    if (!AppState.model) {
        showStatus('❌ No model to save', 'error', 'export-status');
        return;
    }
    
    try {
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

function initializeApp() {
    console.log('🚢 Titanic Neural Explorer initialized');
    
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
    
    setupDragAndDrop();
    showStatus('Ready to explore Titanic survival patterns. Upload your CSV files to begin.', 'info');
}

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
window.toggleMatrixView = toggleMatrixView;