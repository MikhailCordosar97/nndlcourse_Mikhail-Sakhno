// ==================== КОНСТАНТЫ ====================
const SIZE = 16;
let step = 0;
let autoTraining = false;
let autoTimer = null;

// ==================== МОДЕЛИ ====================
let baselineModel, studentModel, inputTensor;

// ==================== DOM ЭЛЕМЕНТЫ ====================
const inputCanvas = document.getElementById('inputCanvas');
const baselineCanvas = document.getElementById('baselineCanvas');
const studentCanvas = document.getElementById('studentCanvas');
const baselineLossDiv = document.getElementById('baselineLoss');
const studentLossDiv = document.getElementById('studentLoss');
const logDiv = document.getElementById('log');
const trainBtn = document.getElementById('trainBtn');
const autoBtn = document.getElementById('autoBtn');
const resetBtn = document.getElementById('resetBtn');

// ==================== ФУНКЦИИ ПОТЕРЬ ====================
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Direction Loss - считаем вручную через массивы
function directionLoss(yPred) {
    return tf.tidy(() => {
        const data = yPred.dataSync();
        let loss = 0;
        
        // Считаем разницу между соседними пикселями по горизонтали
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                loss += (data[idx + 1] - data[idx]); // хотим чтобы справа было светлее
            }
        }
        
        return tf.scalar(-loss / (SIZE * (SIZE - 1))); // минус чтобы минимизировать
    });
}

// Smoothness Loss - штраф за резкие переходы
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const data = yPred.dataSync();
        let loss = 0;
        
        // По горизонтали
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                loss += Math.pow(data[idx + 1] - data[idx], 2);
            }
        }
        
        // По вертикали
        for (let y = 0; y < SIZE - 1; y++) {
            for (let x = 0; x < SIZE; x++) {
                const idx = y * SIZE + x;
                const idx2 = (y + 1) * SIZE + x;
                loss += Math.pow(data[idx2] - data[idx], 2);
            }
        }
        
        return tf.scalar(loss / (SIZE * SIZE * 2));
    });
}

// Student Loss
function studentLoss(yTrue, yPred) {
    const mse = mseLoss(yTrue, yPred);
    const smooth = smoothnessLoss(yPred);
    const dir = directionLoss(yPred);
    
    return mse.add(smooth.mul(1.0)).add(dir.mul(0.5));
}

// ==================== МОДЕЛИ ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        inputShape: [SIZE * SIZE],
        units: SIZE * SIZE,
        activation: 'sigmoid',
        useBias: true
    }));
    
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    
    return model;
}

function createStudentModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        inputShape: [SIZE * SIZE],
        units: 64,
        activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
        units: SIZE * SIZE,
        activation: 'sigmoid'
    }));
    
    return model;
}

// ==================== ВИЗУАЛИЗАЦИЯ ====================
function drawTensor(tensor, canvas) {
    const data = tensor.dataSync();
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cellSize = size / SIZE;
    
    ctx.clearRect(0, 0, size, size);
    
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const val = data[y * SIZE + x];
            const bright = Math.floor(val * 255);
            ctx.fillStyle = `rgb(${bright}, ${bright}, ${bright})`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

function updateDisplays() {
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    const baselinePred = baselineModel.predict(inputTensor);
    const studentPred = studentModel.predict(inputTensor);
    
    const baselineLoss = mseLoss(inputTensor, baselinePred).dataSync()[0];
    const sLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
    
    drawTensor(baselinePred, baselineCanvas);
    drawTensor(studentPred, studentCanvas);
    
    baselineLossDiv.textContent = baselineLoss.toFixed(6);
    studentLossDiv.textContent = sLoss.toFixed(6);
    
    tf.dispose([baselinePred, studentPred]);
}

// ==================== ОБУЧЕНИЕ ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    try {
        // Baseline
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        // Student
        const optimizer = tf.train.adam(0.01);
        optimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, true);
            return studentLoss(inputTensor, pred);
        });
        
        step++;
        
        if (step % 5 === 0) {
            updateDisplays();
            log(`Step ${step}`);
        }
    } catch (e) {
        log('Error: ' + e.message);
    }
}

// ==================== ЛОГ ====================
function log(msg) {
    const time = new Date().toLocaleTimeString();
    logDiv.innerHTML += `<div>[${time}] ${msg}</div>`;
    logDiv.scrollTop = logDiv.scrollHeight;
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================
function init() {
    log('Initializing...');
    
    // Создаем входной шум
    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('Ready! Press Train');
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', trainStep);

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'STOP' : 'AUTO TRAIN';
    autoBtn.className = autoTraining ? 'stop' : '';
    
    if (autoTraining) {
        log('Auto start');
        autoTimer = setInterval(trainStep, 100);
    } else {
        clearInterval(autoTimer);
        log('Stop');
    }
});

resetBtn.addEventListener('click', () => {
    if (autoTraining) {
        clearInterval(autoTimer);
        autoTraining = false;
        autoBtn.textContent = 'AUTO TRAIN';
        autoBtn.className = '';
    }
    
    tf.dispose([baselineModel, studentModel, inputTensor]);
    
    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('Reset');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
