// ==================== КОНСТАНТЫ ====================
const SIZE = 16;
let step = 0;
let autoTraining = false;
let autoTimer = null;

// ==================== МОДЕЛИ ====================
let baselineModel, studentModel, inputTensor, studentOptimizer;

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

// ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
function log(msg) {
    const time = new Date().toLocaleTimeString();
    logDiv.innerHTML += `<div>[${time}] ${msg}</div>`;
    logDiv.scrollTop = logDiv.scrollHeight;
}

function drawTensor(tensor, canvas) {
    let data;
    if (tensor.shape.length === 4) {
        data = tensor.squeeze().dataSync(); // [SIZE, SIZE]
    } else {
        data = tensor.dataSync(); // плоский массив из 256 элементов
    }
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cellSize = size / SIZE;

    ctx.clearRect(0, 0, size, size);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const val = data[y * SIZE + x];
            const bright = Math.floor(val * 255);
            ctx.fillStyle = `rgb(0, ${bright}, 0)`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

// ==================== ФУНКЦИИ ПОТЕРЬ ====================

function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);
        const k = SIZE * SIZE;
        const yTrueSorted = tf.topk(yTrueFlat.neg(), k).values.neg();
        const yPredSorted = tf.topk(yPredFlat.neg(), k).values.neg();
        return tf.mean(tf.square(tf.sub(yTrueSorted, yPredSorted)));
    });
}

function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const yImage = yPred.reshape([1, SIZE, SIZE, 1]);
        const data = yImage.squeeze().dataSync(); // [SIZE, SIZE]
        
        let loss = 0;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                const diff = data[idx + 1] - data[idx];
                loss += diff * diff;
            }
        }
        for (let y = 0; y < SIZE - 1; y++) {
            for (let x = 0; x < SIZE; x++) {
                const idx = y * SIZE + x;
                const idx2 = (y + 1) * SIZE + x;
                const diff = data[idx2] - data[idx];
                loss += diff * diff;
            }
        }
        const totalPairs = (SIZE * (SIZE - 1)) * 2;
        return tf.scalar(loss / totalPairs);
    });
}

function directionLoss(yPred) {
    return tf.tidy(() => {
        // Маска: линейно возрастает по x от 0 до 1
        const range = tf.linspace(0, 1, SIZE); // [SIZE]
        const mask2d = tf.tile(range, [SIZE, 1]); // [SIZE, SIZE]
        const mask = mask2d.reshape([1, SIZE, SIZE, 1]); // [1, SIZE, SIZE, 1]
        
        const yImage = yPred.reshape([1, SIZE, SIZE, 1]);
        return tf.neg(tf.mean(tf.mul(yImage, mask)));
    });
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ПОТЕРЬ ====================
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sorted = sortedMSELoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        // sorted очень маленький – разрешаем переставлять пиксели,
        // smooth – убираем резкость,
        // direction – главный двигатель градиента.
        return sorted
            .mul(0.001)
            .add(smooth.mul(0.05))
            .add(dir.mul(1.0));
    });
}

// ==================== МОДЕЛИ ====================

function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [SIZE * SIZE],
        units: SIZE * SIZE,
        activation: 'sigmoid'
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
        units: 256,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 256,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: SIZE * SIZE,
        activation: 'sigmoid'
    }));
    return model;
}

// ==================== ОБУЧЕНИЕ ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) return;

    try {
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });

        if (!studentOptimizer) studentOptimizer = tf.train.adam(0.01);
        
        studentOptimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, true);
            const loss = studentLoss(inputTensor, pred);
            return loss;
        });

        step++;
        
        if (step % 5 === 0) {
            updateDisplays();
            log(`Step ${step}`);
        }
    } catch (e) {
        log(`Error: ${e.message}`);
        console.error(e);
    }
}

function updateDisplays() {
    tf.tidy(() => {
        const baselinePred = baselineModel.predict(inputTensor);
        const studentPred = studentModel.predict(inputTensor);

        drawTensor(inputTensor, inputCanvas);
        drawTensor(baselinePred, baselineCanvas);
        drawTensor(studentPred, studentCanvas);

        const baselineLoss = mseLoss(inputTensor, baselinePred).dataSync()[0];
        const sLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
        baselineLossDiv.textContent = baselineLoss.toFixed(6);
        studentLossDiv.textContent = sLoss.toFixed(6);
    });
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================
function init() {
    log('Initializing...');

    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);

    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);

    step = 0;
    updateDisplays();
    log('Ready. Press "Train 1 Step" or "Auto Train".');
    log('Sorted=0.001, Smooth=0.05, Direction=1.0');
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', trainStep);

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'STOP' : 'AUTO TRAIN';
    autoBtn.className = autoTraining ? 'stop' : '';
    
    if (autoTraining) {
        log('Auto training started');
        autoTimer = setInterval(trainStep, 30);
    } else {
        clearInterval(autoTimer);
        log('Auto training stopped');
    }
});

resetBtn.addEventListener('click', () => {
    if (autoTraining) {
        clearInterval(autoTimer);
        autoTraining = false;
        autoBtn.textContent = 'AUTO TRAIN';
        autoBtn.className = '';
    }
    
    tf.dispose([baselineModel, studentModel, inputTensor, studentOptimizer]);
    
    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);
    
    step = 0;
    updateDisplays();
    log('Reset done');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
