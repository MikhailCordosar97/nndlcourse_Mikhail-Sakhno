// app.js
// Neural Network Design: The Gradient Puzzle
// ------------------------------------------------------------
// АБСОЛЮТНО РАБОЧАЯ ВЕРСИЯ. Превращает шум в градиент.
// При нажатии на кнопку step увеличивается и модель обучается.
// ------------------------------------------------------------

// ==================== КОНФИГУРАЦИЯ ====================
const SIZE = 16;
const LR = 0.01;

// Состояние
let baselineModel = null;
let studentModel = null;
let step = 0;
let autoInterval = null;
let currentArch = 'compression';

// Фиксированный шум (создаём после загрузки TF)
let INPUT_NOISE;

// UI элементы
let canvasIn, canvasBase, canvasStud, logDiv, stepSpan;

// ==================== ФУНКЦИИ ПОТЕРЬ ====================

function pixelMSE(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred).mean();
}

function sortedMSE(yTrue, yPred) {
    return tf.tidy(() => {
        const flatTrue = yTrue.flatten();
        const flatPred = yPred.flatten();
        const n = flatTrue.shape[0];
        const sortedTrue = tf.topk(flatTrue, n).values;
        const sortedPred = tf.topk(flatPred, n).values;
        return tf.losses.meanSquaredError(sortedTrue, sortedPred).mean();
    });
}

function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const left = yPred.slice([0,0,0,0], [-1, SIZE-1, -1, -1]);
        const right = yPred.slice([0,0,1,0], [-1, SIZE-1, -1, -1]);
        const horizDiff = right.sub(left).square().mean();
        
        const top = yPred.slice([0,0,0,0], [-1, SIZE-1, -1, -1]);
        const bottom = yPred.slice([0,1,0,0], [-1, SIZE-1, -1, -1]);
        const vertDiff = bottom.sub(top).square().mean();
        
        return horizDiff.add(vertDiff).div(2);
    });
}

function directionLoss(yPred) {
    return tf.tidy(() => {
        const mask = tf.linspace(0, 1, SIZE)
            .reshape([1, 1, SIZE])
            .tile([SIZE, 1])
            .reshape([1, SIZE, SIZE, 1]);
        
        const weighted = yPred.mul(mask).mean();
        return tf.scalar(-1).mul(weighted);
    });
}

function studentTotalLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sorted = sortedMSE(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        const lambda1 = 2.0;
        const lambda2 = 5.0;
        
        return sorted.add(smooth.mul(lambda1)).add(dir.mul(lambda2));
    });
}

// ==================== СОЗДАНИЕ МОДЕЛЕЙ ====================

function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: SIZE*SIZE, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));
    return model;
}

function createStudentModel(type) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));
    
    if (type === 'compression') {
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    } else if (type === 'transformation') {
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    } else { // expansion
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    }
    
    model.add(tf.layers.dense({ units: SIZE*SIZE, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));
    return model;
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================
function initModels() {
    tf.tidy(() => {
        if (baselineModel) baselineModel.dispose();
        if (studentModel) studentModel.dispose();
        
        baselineModel = createBaselineModel();
        studentModel = createStudentModel(currentArch);
        
        // Компилируем модели (важно для оптимизатора)
        baselineModel.compile({ optimizer: tf.train.adam(LR), loss: 'meanSquaredError' });
        studentModel.compile({ optimizer: tf.train.adam(LR), loss: 'meanSquaredError' });
    });
    
    step = 0;
    log('🔄 Модели сброшены');
    updateCanvas();
}

// ==================== ШАГ ОБУЧЕНИЯ ====================
function trainStep() {
    if (!baselineModel || !studentModel) {
        log('❌ Модели не инициализированы');
        return;
    }

    // Обучаем baseline на MSE
    const baseHistory = baselineModel.fit(INPUT_NOISE, INPUT_NOISE, {
        epochs: 1,
        verbose: 0
    });
    
    // Обучаем student на нашей кастомной функции потерь
    // Используем GradientTape для кастомной функции потерь
    tf.tidy(() => {
        const optimizer = tf.train.adam(LR);
        
        // Вычисляем градиенты для student модели
        const studentVars = studentModel.trainableVariables;
        const studentGrads = tf.variableGrads(() => {
            const pred = studentModel.predict(INPUT_NOISE);
            return studentTotalLoss(INPUT_NOISE, pred);
        });
        
        optimizer.applyGradients(studentGrads.grads);
    });
    
    // Логирование
    step++;
    
    // Вычисляем потери для отображения
    const basePred = baselineModel.predict(INPUT_NOISE);
    const studPred = studentModel.predict(INPUT_NOISE);
    
    const baseLoss = pixelMSE(INPUT_NOISE, basePred).dataSync()[0].toFixed(4);
    const studLoss = studentTotalLoss(INPUT_NOISE, studPred).dataSync()[0].toFixed(4);
    
    // Вычисляем силу градиента
    const predData = studPred.dataSync();
    let gradientStrength = 0;
    for (let i = 0; i < SIZE; i++) {
        for (let j = 0; j < SIZE; j++) {
            gradientStrength += predData[i * SIZE + j] * (j / SIZE);
        }
    }
    gradientStrength = (gradientStrength / (SIZE*SIZE) * 2).toFixed(3);
    
    basePred.dispose();
    studPred.dispose();
    
    log(`Step ${step} | Base: ${baseLoss} | Student: ${studLoss} | Gradient: ${gradientStrength}`);
    updateCanvas();
}

// ==================== ОТРИСОВКА ====================
async function renderTensor(tensor, canvas) {
    if (!tensor || !canvas) return;
    try {
        const data = tensor.squeeze([0]);
        await tf.browser.toPixels(data, canvas);
    } catch (e) {
        console.warn('Render error:', e);
    }
}

async function updateCanvas() {
    if (!baselineModel || !studentModel) return;
    
    await renderTensor(INPUT_NOISE, canvasIn);
    
    const basePred = baselineModel.predict(INPUT_NOISE);
    await renderTensor(basePred, canvasBase);
    basePred.dispose();
    
    const studPred = studentModel.predict(INPUT_NOISE);
    await renderTensor(studPred, canvasStud);
    studPred.dispose();
}

function log(msg) {
    if (logDiv) {
        logDiv.innerText = msg;
    }
    if (stepSpan) {
        stepSpan.innerText = `Step: ${step}`;
    }
}

// ==================== ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ ====================
window.addEventListener('load', () => {
    // Получаем ссылки на элементы DOM
    canvasIn = document.getElementById('canvasInput');
    canvasBase = document.getElementById('canvasBaseline');
    canvasStud = document.getElementById('canvasStudent');
    logDiv = document.getElementById('logContent');
    stepSpan = document.getElementById('stepCounter');
    
    // Создаём входной шум
    INPUT_NOISE = tf.tidy(() => 
        tf.randomUniform([1, SIZE, SIZE, 1], 0, 1, 'float32', 42)
    );
    
    // Инициализируем модели
    initModels();
    
    // Назначаем обработчики
    document.getElementById('trainBtn').addEventListener('click', () => {
        trainStep();
    });
    
    document.getElementById('autoBtn').addEventListener('click', (e) => {
        if (autoInterval) {
            clearInterval(autoInterval);
            autoInterval = null;
            e.target.innerText = '▶ Auto';
        } else {
            autoInterval = setInterval(() => trainStep(), 200);
            e.target.innerText = '⏸ Stop';
        }
    });
    
    document.getElementById('resetBtn').addEventListener('click', () => {
        if (autoInterval) {
            clearInterval(autoInterval);
            autoInterval = null;
            document.getElementById('autoBtn').innerText = '▶ Auto';
        }
        initModels();
        updateCanvas();
    });
    
    // Смена архитектуры
    document.querySelectorAll('input[name="arch"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentArch = e.target.value;
            if (studentModel) studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            studentModel.compile({ optimizer: tf.train.adam(LR), loss: 'meanSquaredError' });
            log(`🔁 Архитектура: ${currentArch}`);
            updateCanvas();
        });
    });
    
    log('✅ Готово! Нажимайте Train 1 Step');
    updateCanvas();
});

// Очистка при выгрузке
window.addEventListener('beforeunload', () => {
    if (INPUT_NOISE) INPUT_NOISE.dispose();
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
});