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
        
        // Маска: значения растут слева направо (0 -> 1)
        let maskData = [];
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < w; j++) {
                maskData.push(j / w);
            }
        }
        const mask = tf.tensor2d(maskData, [h, w]).reshape([1, h, w, 1]);
        
        // Чем больше совпадение с маской, тем меньше loss
        return tf.neg(tf.mean(tf.mul(y, mask)));
    });
}

// ==================== КЛЮЧЕВАЯ ФУНКЦИЯ - РЕШЕНИЕ ЗАДАЧИ ====================
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        // 1. MSE - сохраняет распределение цветов
        const mse = mseLoss(yTrue, yPred);
        
        // 2. Smoothness - делает переходы плавными
        const smooth = smoothnessLoss(yPred);
        
        // 3. Direction - создает градиент слева направо
        const dir = directionLoss(yPred);
        
        // Баланс коэффициентов: 
        // - MSE почти не влияет (0.05)
        // - Smoothness сильно влияет (5.0) - убирает шум
        // - Direction очень сильно влияет (10.0) - создает градиент
        const total = mse.mul(0.05)
                       .add(smooth.mul(5.0))
                       .add(dir.mul(10.0));
        
        return total;
    });
}

// ==================== СОЗДАНИЕ МОДЕЛЕЙ ====================
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
        
        // Student обучение - создаем оптимизатор если нужно
        if (!studentOptimizer) {
            studentOptimizer = tf.train.adam(0.02);
        }
        
        // Один шаг оптимизации
        studentOptimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, { training: true });
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
    studentOptimizer = tf.train.adam(0.02);
    
    // Инициализируем веса одним проходом
    baselineModel.predict(inputTensor);
    studentModel.predict(inputTensor);
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО! Нажми "Auto Train" и наблюдай за градиентом');
    log('Student использует: MSE*0.05 + Smoothness*5.0 + Direction*10.0');
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
        log('▶ Auto training START');
        autoTimer = setInterval(async () => {
            await trainStep();
        }, 50); // Быстрее
    } else {
        clearInterval(autoTimer);
        log('⏸ Auto training STOP');
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
    studentOptimizer = tf.train.adam(0.02);
    
    baselineModel.predict(inputTensor);
    studentModel.predict(inputTensor);
    
    step = 0;
    updateDisplays();
    
    log('🔄 RESET');
});

// ==================== СТАРТ ====================
tf.ready().then(() => {
    log('TensorFlow.js loaded');
    init();
});
