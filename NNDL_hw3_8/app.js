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

// ==================== ФУНКЦИИ ПОТЕРЬ ====================
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// ==================== ЭТО РЕШЕНИЕ ЗАДАЧИ ====================
// Level 2: Sorted MSE - позволяет перемещать пиксели
function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Расплющиваем в 1D массив
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);
        
        // Сортируем (это позволяет пикселям менять позиции)
        const yTrueSorted = tf.topk(yTrueFlat, SIZE*SIZE).values;
        const yPredSorted = tf.topk(yPredFlat, SIZE*SIZE).values;
        
        // Сравниваем отсортированные последовательности
        return tf.mean(tf.square(tf.sub(yTrueSorted, yPredSorted)));
    });
}

// Level 3: Smoothness - убирает резкие переходы
function smoothnessLoss(y) {
    return tf.tidy(() => {
        const [b, h, w, c] = y.shape;
        
        // Разница по горизонтали
        const left = y.slice([0,0,0,0], [b, h, w-1, c]);
        const right = y.slice([0,0,1,0], [b, h, w-1, c]);
        const hDiff = tf.sub(left, right);
        
        // Разница по вертикали
        const top = y.slice([0,0,0,0], [b, h-1, w, c]);
        const bottom = y.slice([0,1,0,0], [b, h-1, w, c]);
        const vDiff = tf.sub(top, bottom);
        
        return tf.add(
            tf.mean(tf.square(hDiff)),
            tf.mean(tf.square(vDiff))
        );
    });
}

// Level 3: Direction - создает градиент
function directionLoss(y) {
    return tf.tidy(() => {
        const [b, h, w, c] = y.shape;
        
        // Создаем маску: слева 0, справа 1
        const mask = tf.tensor2d(
            Array(h).fill(0).map((_, i) => 
                Array(w).fill(0).map((_, j) => j / w)
            ).flat(),
            [h, w]
        ).reshape([1, h, w, 1]);
        
        // Поощряем соответствие маске
        return tf.neg(tf.mean(tf.mul(y, mask)));
    });
}

// ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ - КАК В ЛЕКЦИИ
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Level 2: Sorted MSE - освобождает пиксели от позиций
        const sortedLoss = sortedMSELoss(yTrue, yPred);
        
        // Level 3: Smoothness - убирает шум
        const smoothLoss = smoothnessLoss(yPred);
        
        // Level 3: Direction - направляет пиксели
        const dirLoss = directionLoss(yPred);
        
        // Комбинация как в лекции
        return sortedLoss
            .add(smoothLoss.mul(0.1))
            .add(dirLoss.mul(0.05));
    });
}

// ==================== МОДЕЛИ ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.dense({
        inputShape: [SIZE*SIZE],
        units: SIZE*SIZE,
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
        inputShape: [SIZE*SIZE],
        units: SIZE*SIZE * 2,
        activation: 'relu'
    }));
    
    model.add(tf.layers.dense({
        units: SIZE*SIZE,
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
        // Baseline обучение
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        // Student обучение
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
    const randomData = new Float32Array(SIZE*SIZE);
    for (let i = 0; i < SIZE*SIZE; i++) {
        randomData[i] = Math.random();
    }
    inputTensor = tf.tensor2d(randomData, [1, SIZE*SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    // Создаем модели
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО! Нажми "Auto Train"');
    log('Sorted MSE + Smoothness + Direction = Градиент');
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
    
    const randomData = new Float32Array(SIZE*SIZE);
    for (let i = 0; i < SIZE*SIZE; i++) {
        randomData[i] = Math.random();
    }
    inputTensor = tf.tensor2d(randomData, [1, SIZE*SIZE]);
    
    drawTensor(inputTensor.reshape([SIZE, SIZE]), inputCanvas);
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel();
    
    step = 0;
    updateDisplays();
    
    log('🔄 Reset');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
