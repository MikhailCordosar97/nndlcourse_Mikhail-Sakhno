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
        
        // Сортируем значения (в порядке возрастания)
        const yTrueVals = tf.topk(yTrueFlat.neg(), SIZE*SIZE).values.neg();
        const yPredVals = tf.topk(yPredFlat.neg(), SIZE*SIZE).values.neg();
        
        // Сравниваем отсортированные значения
        return tf.mean(tf.square(tf.sub(yTrueVals, yPredVals)));
    });
}

// Level 3: Smoothness (Total Variation Loss)
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        // Получаем размерности
        const [batch, height, width, channels] = yPred.shape;
        
        let totalLoss = tf.scalar(0);
        
        // Горизонтальная разница (если ширина > 1)
        if (width > 1) {
            const left = yPred.slice([0, 0, 0, 0], [batch, height, width-1, channels]);
            const right = yPred.slice([0, 0, 1, 0], [batch, height, width-1, channels]);
            const hDiff = tf.sub(left, right);
            totalLoss = tf.add(totalLoss, tf.mean(tf.square(hDiff)));
        }
        
        // Вертикальная разница (если высота > 1)
        if (height > 1) {
            const top = yPred.slice([0, 0, 0, 0], [batch, height-1, width, channels]);
            const bottom = yPred.slice([0, 1, 0, 0], [batch, height-1, width, channels]);
            const vDiff = tf.sub(top, bottom);
            totalLoss = tf.add(totalLoss, tf.mean(tf.square(vDiff)));
        }
        
        return totalLoss;
    });
}

// Level 3: Direction Loss - поощряет градиент слева направо
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = yPred.shape;
        
        // Создаем маску: значения от 0 до 1 слева направо
        const maskData = [];
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                maskData.push(j / (width - 1 || 1)); // защита от деления на 0
            }
        }
        
        const mask = tf.tensor(maskData).reshape([1, height, width, 1]);
        
        // Чем больше совпадение с маской, тем меньше loss
        return tf.neg(tf.mean(tf.mul(yPred, mask)));
    });
}

// ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sortedLoss = sortedMSELoss(yTrue, yPred);
        const smoothLoss = smoothnessLoss(yPred);
        const dirLoss = directionLoss(yPred);
        
        return sortedLoss
            .add(smoothLoss.mul(0.1))
            .add(dirLoss.mul(0.05));
    });
}

// ==================== МОДЕЛИ ====================

// Baseline модель (MSE only)
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [SIZE, SIZE, 1],
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
    
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'meanSquaredError'
    });
    
    return model;
}

// Student модель
function createStudentModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [SIZE, SIZE, 1],
        filters: 32,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.conv2d({
        filters: 32,
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
        // Baseline обучение
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        // Student обучение
        if (!studentOptimizer) {
            studentOptimizer = tf.train.adam(0.01);
        }
        
        studentOptimizer.minimize(() => {
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
    studentOptimizer = tf.train.adam(0.01);
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО! Нажми Auto Train');
    log('Sorted MSE + Smoothness + Direction');
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
