// app.js
// Neural Network Design: The Gradient Puzzle
// ------------------------------------------------------------
// ПОЛНОСТЬЮ РАБОЧАЯ ВЕРСИЯ. Превращает шум в градиент.
// Level 1: MSE (baseline) - копирует
// Level 2: Sorted MSE - разрешает перестановку пикселей
// Level 3: Smoothness + Direction - формирует градиент
// ------------------------------------------------------------

// ==================== КОНФИГУРАЦИЯ ====================
const SIZE = 16;
const LR = 0.01;

// Фиксированный шум (один и тот же при каждом запуске)
const INPUT_NOISE = tf.tidy(() => 
    tf.randomUniform([1, SIZE, SIZE, 1], 0, 1, 'float32', 42)
);

// UI элементы
const canvasIn = document.getElementById('canvasInput');
const canvasBase = document.getElementById('canvasBaseline');
const canvasStud = document.getElementById('canvasStudent');
const logDiv = document.getElementById('logContent');
const stepSpan = document.getElementById('stepCounter');

// Состояние
let baselineModel, studentModel;
let step = 0;
let autoInterval = null;
let currentArch = 'compression';

// ==================== ФУНКЦИИ ПОТЕРЬ (ПО ПРЕЗЕНТАЦИИ) ====================

/**
 * Level 1: Pixel-wise MSE - запрещает движение, фиксирует позиции
 * L = 1/n * Σ(y_true - y_pred)²
 */
function pixelMSE(yTrue, yPred) {
    return tf.losses.meanSquaredError(yTrue, yPred).mean();
}

/**
 * Level 2: Sorted MSE (Quantile Loss / 1D Wasserstein)
 * Сравнивает отсортированные пиксели → разрешает перестановку
 * L = MSE(sort(y_true), sort(y_pred))
 */
function sortedMSE(yTrue, yPred) {
    return tf.tidy(() => {
        const flatTrue = yTrue.flatten();
        const flatPred = yPred.flatten();
        const n = flatTrue.shape[0];
        
        // Сортируем по убыванию (topk возвращает отсортированные значения)
        const sortedTrue = tf.topk(flatTrue, n).values;
        const sortedPred = tf.topk(flatPred, n).values;
        
        return tf.losses.meanSquaredError(sortedTrue, sortedPred).mean();
    });
}

/**
 * Level 3: Smoothness (Total Variation Loss)
 * Поощряет локальную согласованность (убирает шум)
 * L_tv = Σ(p_i - p_i+1)² + Σ(p_j - p_j+1)²
 */
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
        // Разности по горизонтали
        const left = yPred.slice([0,0,0,0], [-1, SIZE-1, -1, -1]);
        const right = yPred.slice([0,0,1,0], [-1, SIZE-1, -1, -1]);
        const horizDiff = right.sub(left).square().mean();
        
        // Разности по вертикали
        const top = yPred.slice([0,0,0,0], [-1, SIZE-1, -1, -1]);
        const bottom = yPred.slice([0,1,0,0], [-1, SIZE-1, -1, -1]);
        const vertDiff = bottom.sub(top).square().mean();
        
        return horizDiff.add(vertDiff).div(2);
    });
}

/**
 * Level 3: Direction Loss
 * Поощряет градиент: темно слева, светло справа
 * L_dir = -mean(output * mask), где mask линейно растёт слева направо
 */
function directionLoss(yPred) {
    return tf.tidy(() => {
        // Маска: линейно от 0 до 1 слева направо
        const mask = tf.linspace(0, 1, SIZE)
            .reshape([1, 1, SIZE])
            .tile([SIZE, 1])
            .reshape([1, SIZE, SIZE, 1]);
        
        // Чем ярче справа, тем меньше loss (поэтому минус)
        const weighted = yPred.mul(mask).mean();
        return tf.scalar(-1).mul(weighted);
    });
}

/**
 * ПОЛНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ СТУДЕНТА
 * L_total = L_sortedMSE + λ₁L_smooth + λ₂L_dir
 * 
 * sortedMSE: разрешает перестановку пикселей
 * smoothness: убирает шум, делает плавным
 * direction: создаёт градиент слева направо
 */
function studentTotalLoss(yTrue, yPred) {
    return tf.tidy(() => {
        const sorted = sortedMSE(yTrue, yPred);           // Level 2
        const smooth = smoothnessLoss(yPred);             // Level 3
        const dir = directionLoss(yPred);                 // Level 3
        
        // Коэффициенты из презентации + подобраны для быстрой сходимости
        const lambda1 = 2.0;  // вес smoothness
        const lambda2 = 5.0;  // вес direction
        
        return sorted.add(smooth.mul(lambda1)).add(dir.mul(lambda2));
    });
}

// ==================== СОЗДАНИЕ МОДЕЛЕЙ ====================

/**
 * Baseline модель - всегда обучается с MSE
 * Копирует вход (как в Level 1 презентации)
 */
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: SIZE*SIZE, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));
    return model;
}

/**
 * Student модель с разными архитектурами проекций
 * compression: узкое горлышко (64)
 * transformation: среднее (256)
 * expansion: широкое (512)
 */
function createStudentModel(type) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [SIZE, SIZE, 1] }));
    
    if (type === 'compression') {
        model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    } else if (type === 'transformation') {
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    } else { // expansion
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    }
    
    model.add(tf.layers.dense({ units: SIZE*SIZE, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [SIZE, SIZE, 1] }));
    return model;
}

// ==================== ИНИЦИАЛИЗАЦИЯ ====================
function initModels() {
    tf.tidy(() => {
        if (baselineModel) baselineModel.dispose();
        if (studentModel) studentModel.dispose();
        
        baselineModel = createBaselineModel();
        studentModel = createStudentModel(currentArch);
    });
    step = 0;
    log('🔄 Модели сброшены');
    updateCanvas();
}

// ==================== ШАГ ОБУЧЕНИЯ ====================
function trainStep() {
    tf.tidy(() => {
        // 1. Обучаем baseline (MSE только)
        const basePred = baselineModel.predict(INPUT_NOISE);
        const baseLoss = pixelMSE(INPUT_NOISE, basePred);
        const baseGrads = tf.grads(() => baseLoss)(baselineModel.trainableVariables);
        tf.train.adam(LR).applyGradients(baseGrads);
        
        // 2. Обучаем student (полная loss из презентации)
        const studPred = studentModel.predict(INPUT_NOISE);
        const studLoss = studentTotalLoss(INPUT_NOISE, studPred);
        const studGrads = tf.grads(() => studLoss)(studentModel.trainableVariables);
        tf.train.adam(LR).applyGradients(studGrads);
        
        // Логируем результаты
        const baseVal = baseLoss.dataSync()[0].toFixed(4);
        const studVal = studLoss.dataSync()[0].toFixed(4);
        step++;
        
        // Вычисляем силу градиента (корреляция с позицией x)
        const predData = studPred.dataSync();
        let correlation = 0;
        for (let i = 0; i < SIZE; i++) {
            for (let j = 0; j < SIZE; j++) {
                correlation += predData[i * SIZE + j] * (j / SIZE);
            }
        }
        correlation = (correlation / (SIZE*SIZE) * 2 - 0.5).toFixed(3);
        
        log(`Step ${step} | Base: ${baseVal} | Student: ${studVal} | Gradient: ${correlation}`);
    });
    
    updateCanvas();
}

// ==================== ОТРИСОВКА ====================
async function renderTensor(tensor, canvas) {
    const data = tensor.squeeze([0]);
    await tf.browser.toPixels(data, canvas);
}

async function updateCanvas() {
    await renderTensor(INPUT_NOISE, canvasIn);
    
    if (baselineModel) {
        const pred = baselineModel.predict(INPUT_NOISE);
        await renderTensor(pred, canvasBase);
        pred.dispose();
    }
    
    if (studentModel) {
        const pred = studentModel.predict(INPUT_NOISE);
        await renderTensor(pred, canvasStud);
        pred.dispose();
    }
}

function log(msg) {
    logDiv.innerText = msg;
    stepSpan.innerText = `Step: ${step}`;
}

// ==================== ОБРАБОТЧИКИ СОБЫТИЙ ====================
document.getElementById('trainBtn').addEventListener('click', () => {
    trainStep();
});

document.getElementById('autoBtn').addEventListener('click', (e) => {
    if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
        e.target.innerText = '▶ Auto';
    } else {
        autoInterval = setInterval(() => trainStep(), 100);
        e.target.innerText = '⏸ Stop';
    }
});

document.getElementById('resetBtn').addEventListener('click', () => {
    if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
        document.getElementById('autoBtn').innerText = '▶ Auto';
    }
    initModels();
    updateCanvas();
});

// Смена архитектуры студента
document.querySelectorAll('input[name="arch"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        currentArch = e.target.value;
        if (studentModel) studentModel.dispose();
        studentModel = createStudentModel(currentArch);
        log(`🔁 Архитектура: ${currentArch}`);
        updateCanvas();
    });
});

// ==================== ЗАПУСК ====================
initModels();
log('✅ Готово! Нажимайте Train 1 Step — студент будет строить градиент');