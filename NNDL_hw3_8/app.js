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

function smoothnessLoss(y) {
    return tf.tidy(() => {
        const [b, h, w, c] = y.shape;
        
        // Горизонтальная разница
        const left = y.slice([0,0,0,0], [b, h, w-1, c]);
        const right = y.slice([0,0,1,0], [b, h, w-1, c]);
        const hDiff = tf.sub(left, right);
        
        // Вертикальная разница
        const top = y.slice([0,0,0,0], [b, h-1, w, c]);
        const bottom = y.slice([0,1,0,0], [b, h-1, w, c]);
        const vDiff = tf.sub(top, bottom);
        
        return tf.add(
            tf.mean(tf.square(hDiff)),
            tf.mean(tf.square(vDiff))
        );
    });
}

function directionLoss(y) {
    return tf.tidy(() => {
        const [b, h, w, c] = y.shape;
        
        // Маска: значения растут слева направо
        const mask = tf.tensor2d(
            Array(h).fill(Array(w).fill(0).map((_, i) => i / w)).flat(),
            [h, w]
        ).reshape([1, h, w, 1]);
        
        return tf.neg(tf.mean(tf.mul(y, mask)));
    });
}

function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mse = mseLoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        return mse.add(smooth.mul(0.2)).add(dir.mul(0.1));
    });
}

// ==================== СОЗДАНИЕ МОДЕЛЕЙ ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [SIZE, SIZE, 1],
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 1,
        kernelSize: 3,
        padding: 'same',
        activation: 'sigmoid'
    }));
    
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });
    
    return model;
}

function createStudentModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [SIZE, SIZE, 1],
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 1,
        kernelSize: 3,
        padding: 'same',
        activation: 'sigmoid'
    }));
    
    return model;
}

// ==================== ВИЗУАЛИЗАЦИЯ ====================
function drawTensor(tensor, canvas) {
    const data = tensor.squeeze().arraySync();
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cellSize = size / SIZE;
    
    ctx.clearRect(0, 0, size, size);
    
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const val = data[y][x];
            const bright = Math.floor(val * 255);
            ctx.fillStyle = `rgb(${bright}, ${bright}, ${bright})`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

function updateDisplays() {
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    tf.tidy(() => {
        const baselinePred = baselineModel.predict(inputTensor);
        const studentPred = studentModel.predict(inputTensor);
        
        const baselineLoss = mseLoss(inputTensor, baselinePred).dataSync()[0];
        const sLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
        
        drawTensor(baselinePred, baselineCanvas);
        drawTensor(studentPred, studentCanvas);
        
        baselineLossDiv.textContent = baselineLoss.toFixed(6);
        studentLossDiv.textContent = sLoss.toFixed(6);
    });
}

// ==================== ОБУЧЕНИЕ ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) {
        log('Models not initialized');
        return;
    }
    
    try {
        // Baseline обучение
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0,
            batchSize: 1
        });
        
        // Student обучение
        const optimizer = tf.train.adam(0.01);
        optimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, true);
            const loss = studentLoss(inputTensor, pred);
            return loss;
        });
        
        step++;
        updateDisplays();
        
        if (step % 10 === 0) {
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
    inputTensor = tf.randomUniform([1, SIZE, SIZE, 1], 0, 1);
    drawTensor(inputTensor, inputCanvas);
    
    // Создаем модели
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    // Инициализируем веса одним проходом
    baselineModel.predict(inputTensor);
    studentModel.predict(inputTensor);
    
    step = 0;
    updateDisplays();
    
    log('Ready! Click "Train 1 Step"');
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', async () => {
    await trainStep();
});

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'Stop' : 'Auto Train';
    autoBtn.className = autoTraining ? 'stop' : '';
    
    if (autoTraining) {
        log('Auto training started');
        autoTimer = setInterval(async () => {
            await trainStep();
        }, 100);
    } else {
        clearInterval(autoTimer);
        log('Auto training stopped');
    }
});

resetBtn.addEventListener('click', () => {
    if (autoTraining) {
        clearInterval(autoTimer);
        autoTraining = false;
        autoBtn.textContent = 'Auto Train';
        autoBtn.className = '';
    }
    
    tf.dispose([baselineModel, studentModel, inputTensor]);
    
    inputTensor = tf.randomUniform([1, SIZE, SIZE, 1], 0, 1);
    drawTensor(inputTensor, inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    baselineModel.predict(inputTensor);
    studentModel.predict(inputTensor);
    
    step = 0;
    updateDisplays();
    
    log('Reset complete');
});

// ==================== СТАРТ ====================
tf.ready().then(() => {
    log('TensorFlow.js loaded');
    init();
});