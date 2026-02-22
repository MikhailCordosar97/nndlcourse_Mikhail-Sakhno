// ==================== КОНСТАНТЫ ====================
const SIZE = 16;
let step = 0;
let autoTraining = false;
let autoTimer = null;
let isTraining = false;

// ==================== МОДЕЛИ ====================
let baselineModel, studentModel, inputTensor;
let baselineOptimizer, studentOptimizer;

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
    const data = tensor.dataSync();
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

// ==================== ФУНКЦИИ ПОТЕРЬ (только тензорные операции) ====================
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function smoothnessLoss(yPred) {
    // yPred форма [1, 256] -> [1, 16, 16, 1]
    const img = yPred.reshape([1, SIZE, SIZE, 1]);

    // Разность по горизонтали
    const left = img.slice([0, 0, 0, 0], [1, SIZE, SIZE-1, 1]);
    const right = img.slice([0, 0, 1, 0], [1, SIZE, SIZE-1, 1]);
    const hDiff = tf.sub(left, right);

    // Разность по вертикали
    const top = img.slice([0, 0, 0, 0], [1, SIZE-1, SIZE, 1]);
    const bottom = img.slice([0, 1, 0, 0], [1, SIZE-1, SIZE, 1]);
    const vDiff = tf.sub(top, bottom);

    return tf.add(tf.mean(tf.square(hDiff)), tf.mean(tf.square(vDiff)));
}

function directionLoss(yPred) {
    // Маска: линейно от 0 до 1 по столбцам
    const maskData = new Float32Array(SIZE * SIZE);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            maskData[y * SIZE + x] = x / (SIZE - 1);
        }
    }
    const mask = tf.tensor(maskData, [1, SIZE * SIZE]);
    // Поощряем корреляцию с маской: чем больше среднее произведение, тем меньше loss
    return tf.neg(tf.mean(tf.mul(yPred, mask)));
}

// Baseline loss (только MSE)
function baselineLoss(yTrue, yPred) {
    return mseLoss(yTrue, yPred);
}

// Student loss (комбинация)
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mse = mseLoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        // Коэффициенты: MSE почти не учитываем, direction — главный
        return mse.mul(0.001)
            .add(smooth.mul(0.01))
            .add(dir.mul(20.0));
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
    if (isTraining) return;

    isTraining = true;
    try {
        // Инициализация оптимизаторов при первом шаге
        if (!baselineOptimizer) baselineOptimizer = tf.train.adam(0.01);
        if (!studentOptimizer) studentOptimizer = tf.train.adam(0.01);

        // Шаг для baseline
        baselineOptimizer.minimize(() => {
            const pred = baselineModel.apply(inputTensor, true);
            const loss = baselineLoss(inputTensor, pred);
            return loss;
        });

        // Шаг для student
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
    } finally {
        isTraining = false;
    }
}

function updateDisplays() {
    tf.tidy(() => {
        const baselinePred = baselineModel.predict(inputTensor);
        const studentPred = studentModel.predict(inputTensor);

        drawTensor(inputTensor, inputCanvas);
        drawTensor(baselinePred, baselineCanvas);
        drawTensor(studentPred, studentCanvas);

        const baselineLossVal = baselineLoss(inputTensor, baselinePred).dataSync()[0];
        const studentLossVal = studentLoss(inputTensor, studentPred).dataSync()[0];
        baselineLossDiv.textContent = baselineLossVal.toFixed(6);
        studentLossDiv.textContent = studentLossVal.toFixed(6);
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

    step = 0;
    updateDisplays();
    log('Ready. Press "Train 1 Step" or "Auto Train".');
    log('MSE=0.001, Smooth=0.01, Direction=20.0');
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', trainStep);

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'STOP' : 'AUTO TRAIN';
    autoBtn.className = autoTraining ? 'stop' : '';

    if (autoTraining) {
        log('Auto training started');
        autoTimer = setInterval(() => {
            trainStep();
        }, 200);
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

    tf.dispose([baselineModel, studentModel, inputTensor, baselineOptimizer, studentOptimizer]);

    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);

    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    baselineOptimizer = null;
    studentOptimizer = null;

    step = 0;
    updateDisplays();
    log('Reset done');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
