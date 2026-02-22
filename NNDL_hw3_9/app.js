// ==================== КОНСТАНТЫ ====================
const SIZE = 16;
let step = 0;
let autoTraining = false;
let autoTimer = null;
let isTraining = false;

// ==================== МОДЕЛИ ====================
let studentModel, inputTensor, studentOptimizer;

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

// ==================== ФУНКЦИИ ПОТЕРЬ ====================
// Без TopK, без сортировки. Только гладкость и направление.
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const flat = yPred.dataSync();
        let loss = 0;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                const diff = flat[idx + 1] - flat[idx];
                loss += diff * diff;
            }
        }
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

function directionLoss(yPred) {
    return tf.tidy(() => {
        const flat = yPred.dataSync();
        let loss = 0;
        for (let y = 0; y < SIZE; y++) {
            for (let x = 0; x < SIZE - 1; x++) {
                const idx = y * SIZE + x;
                const diff = flat[idx + 1] - flat[idx];
                // Если следующий пиксель темнее предыдущего – штраф
                if (diff < 0) {
                    loss -= diff; // так как diff отрицательный, -diff положительный
                }
            }
        }
        return tf.scalar(loss / (SIZE * (SIZE - 1)));
    });
}

function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mse = mseLoss(yTrue, yPred);
        const smooth = smoothnessLoss(yPred);
        const dir = directionLoss(yPred);
        
        // Баланс: mse маленький – разрешаем менять цвета,
        // smooth – убирает резкость,
        // dir – большой, чтобы создать градиент.
        return mse
            .mul(0.01)
            .add(smooth.mul(0.1))
            .add(dir.mul(20.0));
    });
}

// ==================== МОДЕЛИ ====================
// Убираем baseline модель полностью. Оставляем только студента.
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
    if (!inputTensor || !studentModel) return;
    if (isTraining) return;

    isTraining = true;
    try {
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
    } finally {
        isTraining = false;
    }
}

function updateDisplays() {
    tf.tidy(() => {
        const studentPred = studentModel.predict(inputTensor);

        drawTensor(inputTensor, inputCanvas);
        drawTensor(studentPred, studentCanvas);
        // Для baseline просто показываем тот же input (или можно оставить пустым)
        drawTensor(inputTensor, baselineCanvas);

        const sLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
        baselineLossDiv.textContent = '0.000000'; // baseline не обучается
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

    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);

    step = 0;
    updateDisplays();
    log('Ready. Press "Train 1 Step" or "Auto Train".');
    log('MSE=0.01, Smooth=0.1, Direction=20.0 (без сортировки)');
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
        }, 300); // интервал 300 мс, чтобы избежать пересечений
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
    
    tf.dispose([studentModel, inputTensor, studentOptimizer]);
    
    const data = new Float32Array(SIZE * SIZE);
    for (let i = 0; i < data.length; i++) {
        data[i] = Math.random();
    }
    inputTensor = tf.tensor2d(data, [1, SIZE * SIZE]);
    
    studentModel = createStudentModel();
    studentOptimizer = tf.train.adam(0.01);
    
    step = 0;
    updateDisplays();
    log('Reset done');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
