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
        data = tensor.squeeze().dataSync();
    } else {
        data = tensor.dataSync();
    }
    const ctx = canvas.getContext('2d');
    const size = canvas.width;
    const cellSize = size / SIZE;

    ctx.clearRect(0, 0, size, size);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            const val = data[y * SIZE + x];
            const bright = Math.floor(val * 255);
            // Зеленые оттенки
            ctx.fillStyle = `rgb(0, ${bright}, 0)`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

// ==================== ФУНКЦИИ ПОТЕРЬ ====================

// Обычная MSE (для baseline)
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Sorted MSE – сравниваем отсортированные пиксели (разрешает перестановку)
function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);
        const k = SIZE * SIZE;
        // Сортируем по возрастанию через отрицание
        const yTrueSorted = tf.topk(yTrueFlat.neg(), k).values.neg();
        const yPredSorted = tf.topk(yPredFlat.neg(), k).values.neg();
        return tf.mean(tf.square(tf.sub(yTrueSorted, yPredSorted)));
    });
}

// Smoothness – штраф за резкие перепады (total variation)
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        let flat;
        if (yPred.shape.length === 4) {
            flat = yPred.squeeze().dataSync();
        } else {
            flat = yPred.dataSync();
        }
        
        let loss = 0;
        // горизонтальные разности
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                const diff = flat[idx + 1] - flat[idx];
                loss += diff * diff;
            }
        }
        // вертикальные разности
        for (let y = 0; y < SIZE - 1; y++) {
            for (let x = 0; x < SIZE; x++) {
                const idx = y * SIZE + x;
                const idx2 = (y + 1) * SIZE + x;
                const diff = flat[idx2] - flat[idx];
                loss += diff * diff;
            }
        }
        const totalPairs = (SIZE * (SIZE - 1)) * 2;
        return tf.scalar(loss / totalPairs);
    });
}

// Direction Loss – поощряет градиент слева направо (маска от 0 до 1)
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width] = yPred.shape;
        // Создаём маску: значения растут с x
        const mask = [];
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                mask.push(j / (width - 1));
            }
        }
        const maskTensor = tf.tensor(mask).reshape([1, height, width]);
        // Поощряем совпадение: чем больше среднее произведение, тем меньше loss
        // Используем отрицательный знак, так как минимизируем
        return tf.neg(tf.mean(tf.mul(yPred, maskTensor)));
    });
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ПОТЕРЬ ====================
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sorted = sortedMSELoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        // КЛЮЧЕВЫЕ КОЭФФИЦИЕНТЫ:
        // sorted очень маленький – разрешаем менять цвета,
        // smooth средний – убираем шум,
        // direction большой – заставляем пиксели выстраиваться в градиент.
        return sorted
            .mul(0.001)
            .add(smooth.mul(0.05))
            .add(dir.mul(1.0));
    });
}

// ==================== МОДЕЛИ ====================

// Baseline – просто копирует вход (MSE)
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

// Student – достаточно мощная, чтобы переставлять пиксели
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
        // Baseline обучается на MSE
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });

        // Student обучается с нашей кастомной loss
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

    // Случайный шум
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
        autoTimer = setInterval(trainStep, 30); // быстрее
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
