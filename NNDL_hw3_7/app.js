// ==================== Глобальные переменные ====================
let inputTensor, baselineModel, studentModel, optimizer;
let step = 0;
let isAutoTraining = false;
let animationFrame = null;
const IMG_SIZE = 16;

// DOM элементы
const inputCanvas = document.getElementById('inputCanvas');
const baselineCanvas = document.getElementById('baselineCanvas');
const studentCanvas = document.getElementById('studentCanvas');
const baselineLossSpan = document.getElementById('baselineLossVal');
const studentLossSpan = document.getElementById('studentLossVal');
const stepSpan = document.getElementById('stepCount');
const logArea = document.getElementById('logArea');
const trainOneBtn = document.getElementById('trainOneBtn');
const autoTrainBtn = document.getElementById('autoTrainBtn');
const resetBtn = document.getElementById('resetBtn');
const archRadios = document.querySelectorAll('input[name="arch"]');

// ==================== Функции потерь ====================
function mse(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Smoothness Loss
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = yPred.shape;
        
        // По горизонтали
        const left = yPred.slice([0, 0, 0, 0], [batch, height, width-1, channels]);
        const right = yPred.slice([0, 0, 1, 0], [batch, height, width-1, channels]);
        const horizontalDiff = tf.sub(left, right);
        
        // По вертикали
        const top = yPred.slice([0, 0, 0, 0], [batch, height-1, width, channels]);
        const bottom = yPred.slice([0, 1, 0, 0], [batch, height-1, width, channels]);
        const verticalDiff = tf.sub(top, bottom);
        
        return tf.add(
            tf.mean(tf.square(horizontalDiff)),
            tf.mean(tf.square(verticalDiff))
        );
    });
}

// Direction Loss
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = yPred.shape;
        
        // Маска слева-направо
        let mask = [];
        for (let i = 0; i < width; i++) {
            mask.push(i / width);
        }
        
        const maskTensor = tf.tensor2d(
            Array(height).fill(mask).flat(),
            [height, width]
        ).reshape([1, height, width, 1]);
        
        return tf.neg(tf.mean(tf.mul(yPred, maskTensor)));
    });
}

// Student loss
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const mseLoss = mse(yTrue, yPred);
        const smoothLoss = smoothnessLoss(yPred);
        const dirLoss = directionLoss(yPred);
        
        return mseLoss.add(smoothLoss.mul(0.2)).add(dirLoss.mul(0.1));
    });
}

// ==================== Создание моделей ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [IMG_SIZE, IMG_SIZE, 1],
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

function createStudentModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [IMG_SIZE, IMG_SIZE, 1],
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

// ==================== Инициализация ====================
async function init() {
    log('Initializing...');
    
    // Создаем входной шум
    inputTensor = tf.randomUniform([1, IMG_SIZE, IMG_SIZE, 1], 0, 1);
    
    // Рисуем вход
    drawTensor(inputTensor, inputCanvas);
    
    // Создаем модели
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    // Компилируем baseline
    baselineModel.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });
    
    // Оптимизатор для student
    optimizer = tf.train.adam(0.01);
    
    step = 0;
    stepSpan.textContent = step;
    
    // Первое предсказание
    updateDisplays();
    
    log('✅ Ready. Press "Train 1 Step" to begin.');
}

// ==================== Обучение ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    try {
        // Baseline обучение
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0,
            batchSize: 1
        });
        
        // Student обучение
        optimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, { training: true });
            const loss = studentLoss(inputTensor, pred);
            return loss;
        });
        
        step++;
        stepSpan.textContent = step;
        
        updateDisplays();
        
        if (step % 10 === 0) {
            log(`Step ${step} completed`);
        }
    } catch (error) {
        log(`Error: ${error.message}`);
        console.error(error);
    }
}

// ==================== Визуализация ====================
function drawTensor(tensor, canvas) {
    const data = tensor.squeeze().dataSync();
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const cellSize = width / IMG_SIZE;
    
    ctx.clearRect(0, 0, width, height);
    
    for (let y = 0; y < IMG_SIZE; y++) {
        for (let x = 0; x < IMG_SIZE; x++) {
            const val = data[y * IMG_SIZE + x];
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
    
    const baselineLoss = mse(inputTensor, baselinePred).dataSync()[0];
    const studentLoss = studentLoss(inputTensor, studentPred).dataSync()[0];
    
    drawTensor(baselinePred, baselineCanvas);
    drawTensor(studentPred, studentCanvas);
    
    baselineLossSpan.textContent = baselineLoss.toFixed(6);
    studentLossSpan.textContent = studentLoss.toFixed(6);
    
    tf.dispose([baselinePred, studentPred]);
}

function log(msg) {
    const time = new Date().toLocaleTimeString();
    logArea.innerHTML += `<div>[${time}] ${msg}</div>`;
    logArea.scrollTop = logArea.scrollHeight;
}

// ==================== Управление ====================
trainOneBtn.addEventListener('click', trainStep);

autoTrainBtn.addEventListener('click', () => {
    if (isAutoTraining) {
        isAutoTraining = false;
        autoTrainBtn.textContent = '▶ Auto Train';
        autoTrainBtn.classList.remove('stop');
        cancelAnimationFrame(animationFrame);
        log('Auto training stopped');
    } else {
        isAutoTraining = true;
        autoTrainBtn.textContent = '⏸ Stop';
        autoTrainBtn.classList.add('stop');
        
        function loop() {
            if (!isAutoTraining) return;
            trainStep().then(() => {
                animationFrame = requestAnimationFrame(loop);
            });
        }
        
        animationFrame = requestAnimationFrame(loop);
        log('Auto training started');
    }
});

resetBtn.addEventListener('click', () => {
    // Останавливаем автообучение
    if (isAutoTraining) {
        isAutoTraining = false;
        autoTrainBtn.textContent = '▶ Auto Train';
        autoTrainBtn.classList.remove('stop');
        cancelAnimationFrame(animationFrame);
    }
    
    // Очищаем старые модели
    tf.dispose([baselineModel, studentModel]);
    
    // Создаем новые
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    baselineModel.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });
    
    optimizer = tf.train.adam(0.01);
    step = 0;
    stepSpan.textContent = step;
    
    updateDisplays();
    log('Models reset');
});

// ==================== Запуск ====================
tf.ready().then(init);
