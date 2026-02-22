// ==================== КОНСТАНТЫ ====================
const SIZE = 16;
let step = 0;
let autoTraining = false;
let autoTimer = null;

// ==================== МОДЕЛИ ====================
let baselineModel, studentModel, inputTensor;
let studentOptimizer;

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

// ==================== ФУНКЦИИ ПОТЕРЬ ИЗ ЛЕКЦИИ ====================

// Level 1: Обычная MSE
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Level 2: Sorted MSE - позволяет пикселям перемещаться
function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Расплющиваем в 1D массив
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);
        
        // Сортируем значения
        const yTrueVals = tf.topk(yTrueFlat, SIZE*SIZE).values;
        const yPredVals = tf.topk(yPredFlat, SIZE*SIZE).values;
        
        // Сравниваем отсортированные значения
        return tf.mean(tf.square(tf.sub(yTrueVals, yPredVals)));
    });
}

// Level 3: Smoothness - через свертки (без slice)
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        // Ядра для разности соседей
        const kernelH = tf.tensor4d([[[[-1]]], [[[1]]]], [2, 1, 1, 1]);
        const kernelV = tf.tensor4d([[[[-1], [1]]]], [1, 2, 1, 1]);
        
        // Свертки для вычисления разностей
        const hDiff = tf.depthwiseConv2d(yPred, kernelH, 1, 'same');
        const vDiff = tf.depthwiseConv2d(yPred, kernelV, 1, 'same');
        
        return tf.add(
            tf.mean(tf.square(hDiff)),
            tf.mean(tf.square(vDiff))
        );
    });
}

// Level 3: Direction Loss
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = yPred.shape;
        
        // Создаем маску: значения от 0 до 1 слева направо
        const maskData = [];
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                maskData.push(j / (width - 1));
            }
        }
        const mask = tf.tensor(maskData).reshape([1, height, width, 1]);
        
        // Поощряем соответствие маске
        return tf.neg(tf.mean(tf.mul(yPred, mask)));
    });
}

// ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sortedLoss = sortedMSELoss(yTrue, yPred);
        const smoothLoss = smoothnessLoss(yPred);
        const dirLoss = directionLoss(yPred);
        
        // Коэффициенты как в лекции
        return sortedLoss
            .add(smoothLoss.mul(0.1))
            .add(dirLoss.mul(0.05));
    });
}

// ==================== МОДЕЛИ ====================

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
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    
    return model;
}

function createStudentModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [SIZE, SIZE, 1],
        filters: 16,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 16,
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
    const data = tensor.squeeze().dataSync();
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
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    try {
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        if (!studentOptimizer) {
            studentOptimizer = tf.train.adam(0.01);
        }
        
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
    
    inputTensor = tf.randomUniform([1, SIZE, SIZE, 1], 0, 1);
    drawTensor(inputTensor, inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО! Нажми Auto Train');
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
    
    inputTensor = tf.randomUniform([1, SIZE, SIZE, 1], 0, 1);
    drawTensor(inputTensor, inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);
    
    step = 0;
    updateDisplays();
    
    log('🔄 Reset');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
