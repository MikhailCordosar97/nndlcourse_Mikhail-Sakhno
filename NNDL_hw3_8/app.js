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

// ==================== ПРОСТЫЕ ФУНКЦИИ ПОТЕРЬ ====================
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Direction Loss - простой градиент
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [b, h, w, c] = yPred.shape;
        let loss = tf.scalar(0);
        
        // Поощряем увеличение яркости слева направо
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w-1; x++) {
                const left = yPred.slice([0, y, x, 0], [1, 1, 1, 1]);
                const right = yPred.slice([0, y, x+1, 0], [1, 1, 1, 1]);
                loss = tf.add(loss, tf.sub(left, right));
            }
        }
        
        return tf.div(loss, (h * (w-1)));
    });
}

// Smoothness Loss - простой
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const [b, h, w, c] = yPred.shape;
        let loss = tf.scalar(0);
        
        // Штрафуем за резкие переходы
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w-1; x++) {
                const left = yPred.slice([0, y, x, 0], [1, 1, 1, 1]);
                const right = yPred.slice([0, y, x+1, 0], [1, 1, 1, 1]);
                loss = tf.add(loss, tf.square(tf.sub(left, right)));
            }
        }
        
        for (let y = 0; y < h-1; y++) {
            for (let x = 0; x < w; x++) {
                const top = yPred.slice([0, y, x, 0], [1, 1, 1, 1]);
                const bottom = yPred.slice([0, y+1, x, 0], [1, 1, 1, 1]);
                loss = tf.add(loss, tf.square(tf.sub(top, bottom)));
            }
        }
        
        return tf.div(loss, (h * w * 2));
    });
}

// Student Loss - комбинация
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mse = mseLoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        return mse
            .add(smooth.mul(2.0))
            .add(dir.mul(1.0));
    });
}

// ==================== ПРОСТЫЕ МОДЕЛИ ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        inputShape: [SIZE * SIZE],
        units: SIZE * SIZE,
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
    
    model.add(tf.layers.dense({
        inputShape: [SIZE * SIZE],
        units: 128,
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
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
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
    const randomData = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
        randomData[i] = Math.random();
    }
    inputTensor = tf.tensor2d(randomData, [1, SIZE * SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО!');
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', async () => {
    await trainStep();
});

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'STOP' : 'AUTO TRAIN';
    autoBtn.className = autoTraining ? 'stop' : '';
    
    if (autoTraining) {
        log('▶ Auto training');
        autoTimer = setInterval(async () => {
            await trainStep();
        }, 100);
    } else {
        clearInterval(autoTimer);
        log('⏸ Stopped');
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
    
    const randomData = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
        randomData[i] = Math.random();
    }
    inputTensor = tf.tensor2d(randomData, [1, SIZE * SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('🔄 Reset');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
