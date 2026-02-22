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

// Отрисовка тензора [1, SIZE, SIZE, 1] или [1, SIZE*SIZE] на canvas
function drawTensor(tensor, canvas) {
    let data;
    if (tensor.shape.length === 4) {
        data = tensor.squeeze().dataSync();
    } else {
        // [1, SIZE*SIZE]
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
            ctx.fillStyle = `rgb(${bright}, ${bright}, ${bright})`;
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
}

// ==================== ФУНКЦИИ ПОТЕРЬ ====================

// MSE (для baseline)
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Sorted MSE (сравнение распределений)
function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Расплющиваем в 1D
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);

        // Сортируем по возрастанию (через отрицание)
        const k = SIZE * SIZE;
        const yTrueSorted = tf.topk(yTrueFlat.neg(), k).values.neg();
        const yPredSorted = tf.topk(yPredFlat.neg(), k).values.neg();

        return tf.mean(tf.square(tf.sub(yTrueSorted, yPredSorted)));
    });
}

// Smoothness Loss через обычные циклы (надёжно)
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const data = yPred.dataSync(); // [1, SIZE*SIZE] или [SIZE,SIZE]? 
        // yPred может быть [1, SIZE*SIZE] или [1, SIZE, SIZE, 1]. Приведём к плоскому массиву.
        let flat;
        if (yPred.shape.length === 4) {
            flat = yPred.squeeze().dataSync();
        } else {
            flat = yPred.dataSync();
        }
        
        let loss = 0;
        // Горизонтальные разности
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                const diff = flat[idx + 1] - flat[idx];
                loss += diff * diff;
            }
        }
        // Вертикальные разности
        for (let y = 0; y < SIZE - 1; y++) {
            for (let x = 0; x < SIZE; x++) {
                const idx = y * SIZE + x;
                const idx2 = (y + 1) * SIZE + x;
                const diff = flat[idx2] - flat[idx];
                loss += diff * diff;
            }
        }
        const totalPairs = (SIZE * (SIZE - 1)) * 2; // горизонтальные + вертикальные
        return tf.scalar(loss / totalPairs);
    });
}

// Direction Loss (чем правее, тем ярче)
function directionLoss(yPred) {
    return tf.tidy(() => {
        let flat;
        if (yPred.shape.length === 4) {
            flat = yPred.squeeze().dataSync();
        } else {
            flat = yPred.dataSync();
        }
        
        let loss = 0;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE; x++) {
                // желаемое значение пропорционально x
                const target = x / (SIZE - 1);
                loss += (flat[y * SIZE + x] - target) ** 2;
            }
        }
        return tf.scalar(loss / (SIZE * SIZE));
    });
}

// ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ СТУДЕНТА (как в лекции)
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sorted = sortedMSELoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        // Коэффициенты можно подбирать
        return sorted
            .add(smooth.mul(0.1))
            .add(dir.mul(0.05));
    });
}

// ==================== МОДЕЛИ ====================

// Baseline: простая полносвязная (MSE)
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

// Student: тоже полносвязная, но с кастомной loss
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

// ==================== ОБУЧЕНИЕ ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) return;

    try {
        // Baseline обучается на MSE
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });

        // Student обучается через оптимизатор с нашей кастомной loss
        if (!studentOptimizer) studentOptimizer = tf.train.adam(0.01);
        
        studentOptimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, true);
            const loss = studentLoss(inputTensor, pred);
            return loss;
        });

        step++;
        
        // Обновляем экран каждые 5 шагов
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

        // Для визуализации преобразуем в изображение SIZE x SIZE
        drawTensor(inputTensor, inputCanvas);
        drawTensor(baselinePred, baselineCanvas);
        drawTensor(studentPred, studentCanvas);

        // Покажем значения loss (для информации)
        const baselineLoss = mseLoss(inputTensor, baselinePred).dataSync()[0];
        const sLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
        baselineLossDiv.textContent = baselineLoss.toFixed(6);
        studentLossDiv.textContent = sLoss.toFixed(6);
    });
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================
function init() {
    log('Initializing...');

    // Случайный шум [0,1]
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
}

// ==================== ОБРАБОТЧИКИ ====================
trainBtn.addEventListener('click', trainStep);

autoBtn.addEventListener('click', () => {
    autoTraining = !autoTraining;
    autoBtn.textContent = autoTraining ? 'STOP' : 'AUTO TRAIN';
    autoBtn.className = autoTraining ? 'stop' : '';
    
    if (autoTraining) {
        log('Auto training started');
        autoTimer = setInterval(trainStep, 50);
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
