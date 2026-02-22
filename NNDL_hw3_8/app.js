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

// Level 1: Обычная MSE (попиксельное сравнение) - приводит к копированию
function mseLoss(yTrue, yPred) {
    return tf.mean(tf.square(tf.sub(yTrue, yPred)));
}

// Level 2: Sorted MSE (Quantile Loss / 1D Wasserstein)
// КЛЮЧЕВАЯ ИДЕЯ: сравниваем ОТСОРТИРОВАННЫЕ пиксели
function sortedMSELoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Расплющиваем в 1D
        const yTrueFlat = yTrue.reshape([-1]);
        const yPredFlat = yPred.reshape([-1]);
        
        // СОРТИРУЕМ - это позволяет пикселям перемещаться
        const yTrueSorted = tf.topk(yTrueFlat, SIZE*SIZE).values;
        const yPredSorted = tf.topk(yPredFlat, SIZE*SIZE).values;
        
        // Сравниваем отсортированные последовательности
        return tf.mean(tf.square(tf.sub(yTrueSorted, yPredSorted)));
    });
}

// Level 3: Smoothness (Total Variation Loss)
// "Be locally consistent" - убирает резкие переходы
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        // Разница между соседними пикселями по горизонтали
        const left = yPred.slice([0, 0, 0, 0], [1, SIZE, SIZE-1, 1]);
        const right = yPred.slice([0, 0, 1, 0], [1, SIZE, SIZE-1, 1]);
        const horizontalDiff = tf.sub(left, right);
        
        // Разница между соседними пикселями по вертикали
        const top = yPred.slice([0, 0, 0, 0], [1, SIZE-1, SIZE, 1]);
        const bottom = yPred.slice([0, 1, 0, 0], [1, SIZE-1, SIZE, 1]);
        const verticalDiff = tf.sub(top, bottom);
        
        // Сумма квадратов разниц
        return tf.add(
            tf.mean(tf.square(horizontalDiff)),
            tf.mean(tf.square(verticalDiff))
        );
    });
}

// Level 3: Direction Loss
// "Be bright on the right" - поощряет градиент
function directionLoss(yPred) {
    return tf.tidy(() => {
        // Создаем маску: слева темно (0), справа светло (1)
        const mask = [];
        for (let i = 0; i < SIZE; i++) {
            const row = [];
            for (let j = 0; j < SIZE; j++) {
                row.push(j / SIZE); // значение растет слева направо
            }
            mask.push(row);
        }
        const maskTensor = tf.tensor(mask).reshape([1, SIZE, SIZE, 1]);
        
        // Чем больше совпадение с маской, тем меньше loss
        return tf.neg(tf.mean(tf.mul(yPred, maskTensor)));
    });
}

// ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ КАК В ЛЕКЦИИ:
// L_total = L_sortedMSE + λ1 * L_smooth + λ2 * L_dir
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sortedLoss = sortedMSELoss(yTrue, yPred);
        const smoothLoss = smoothnessLoss(yPred);
        const dirLoss = directionLoss(yPred);
        
        // Коэффициенты из лекции (подобраны для градиента)
        const lambda1 = 0.1; // Smoothness
        const lambda2 = 0.05; // Direction
        
        return sortedLoss
            .add(smoothLoss.mul(lambda1))
            .add(dirLoss.mul(lambda2));
    });
}

// ==================== МОДЕЛИ ====================

// Baseline модель (MSE only)
function createBaselineModel() {
    const model = tf.sequential();
    
    // Простая CNN как в лекции
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
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError'
    });
    
    return model;
}

// Student модель (будет учиться с кастомной loss)
function createStudentModel() {
    const model = tf.sequential();
    
    // Та же архитектура что и у baseline
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
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    try {
        // Baseline обучение (MSE)
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        // Student обучение (кастомная loss)
        if (!studentOptimizer) {
            studentOptimizer = tf.train.adam(0.001);
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
    studentOptimizer = tf.train.adam(0.001);
    
    step = 0;
    updateDisplays();
    
    log('ГОТОВО! Нажми Auto Train');
    log('Sorted MSE освобождает пиксели от позиций');
    log('Smoothness + Direction направляют их в градиент');
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
        }, 50);
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
    studentOptimizer = tf.train.adam(0.001);
    
    step = 0;
    updateDisplays();
    
    log('🔄 Reset');
});

// ==================== СТАРТ ====================
tf.ready().then(init);
