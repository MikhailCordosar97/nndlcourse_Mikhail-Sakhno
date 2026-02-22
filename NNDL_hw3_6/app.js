// ==================== Глобальные переменные ====================
let inputTensor, baselineModel, studentModel, optimizer;
let step = 0;
let isAutoTraining = false;
let animationFrame = null;
const IMG_SIZE = 16;
const SMOOTHNESS_COEF = 0.1;
const DIRECTION_COEF = 0.05;

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

// Smoothness Loss (Total Variation) - поощряет плавность переходов
function smoothnessLoss(yPred) {
    const [batch, height, width, channels] = yPred.shape;
    
    // Разница по горизонтали
    const left = yPred.slice([0, 0, 0, 0], [batch, height, width-1, channels]);
    const right = yPred.slice([0, 0, 1, 0], [batch, height, width-1, channels]);
    const horizontalDiff = tf.sub(left, right);
    
    // Разница по вертикали
    const top = yPred.slice([0, 0, 0, 0], [batch, height-1, width, channels]);
    const bottom = yPred.slice([0, 1, 0, 0], [batch, height-1, width, channels]);
    const verticalDiff = tf.sub(top, bottom);
    
    const horizontalLoss = tf.mean(tf.square(horizontalDiff));
    const verticalLoss = tf.mean(tf.square(verticalDiff));
    
    return tf.add(horizontalLoss, verticalLoss);
}

// Direction Loss - поощряет градиент слева направо
function directionLoss(yPred) {
    const [batch, height, width, channels] = yPred.shape;
    
    // Создаем маску: значения увеличиваются слева направо
    const mask = tf.tidy(() => {
        const range = tf.range(0, 1, 1/width);
        const mask2d = tf.tile(range.reshape([1, width]), [height, 1]);
        return mask2d.reshape([1, height, width, 1]);
    });
    
    // Наказываем за несоответствие маске (чем меньше значение, тем лучше)
    // Используем отрицательный коэффициент, чтобы поощрять соответствие
    const loss = tf.neg(tf.mean(tf.mul(yPred, mask)));
    mask.dispose();
    return loss;
}

// ==================== Функции потерь для студентов (TODO-B РЕШЕНО) ====================
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Базовая MSE потеря
        const baseLoss = mse(yTrue, yPred);
        
        // Добавляем smoothness для плавности
        const smoothLoss = smoothnessLoss(yPred);
        
        // Добавляем direction для градиента
        const dirLoss = directionLoss(yPred);
        
        // Комбинируем с коэффициентами
        const total = tf.add(
            baseLoss,
            tf.mul(smoothness(yPred), SMOOTHNESS_COEF),
            tf.mul(directionLoss(yPred), DIRECTION_COEF)
        );
        
        return total;
    });
}

// ==================== Создание моделей ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    // Encoder (Compression)
    model.add(tf.layers.conv2d({
        inputShape: [IMG_SIZE, IMG_SIZE, 1],
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    
    // Bottleneck
    model.add(tf.layers.conv2d({
        filters: 4,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    // Decoder (Expansion)
    model.add(tf.layers.upSampling2d({ size: 2 }));
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
        optimizer: 'adam',
        loss: 'meanSquaredError'
    });
    
    return model;
}

// TODO-A: Архитектуры для студенческой модели (РЕШЕНО)
function createStudentModel(archType) {
    const model = tf.sequential();
    
    switch(archType) {
        case 'compression':
            // Компрессия: уменьшаем размерность, потом восстанавливаем
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
                filters: 16,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
            model.add(tf.layers.conv2d({
                filters: 8,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.upSampling2d({ size: 2 }));
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
            break;
            
        case 'transformation':
            // Трансформация: сохраняем размерность, меняем представление
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
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
            // Residual connection (имитация)
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
            break;
            
        case 'expansion':
            // Расширение: увеличиваем размерность для богатого представления
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
                filters: 64,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.conv2d({
                filters: 128,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.conv2d({
                filters: 256,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            // Сжимаем обратно
            model.add(tf.layers.conv2d({
                filters: 64,
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
            break;
    }
    
    // Не компилируем здесь - будем использовать custom training loop
    return model;
}

// ==================== Инициализация ====================
async function init() {
    // Создаем фиксированный шум
    inputTensor = tf.tidy(() => {
        return tf.randomUniform([1, IMG_SIZE, IMG_SIZE, 1], 0, 1);
    });
    
    // Рисуем вход
    drawTensorToCanvas(inputTensor, inputCanvas);
    
    // Создаем модели
    baselineModel = createBaselineModel();
    studentModel = createStudentModel('compression');
    
    // Оптимизатор для кастомного обучения
    optimizer = tf.train.adam(0.01);
    
    step = 0;
    updateStepDisplay();
    log('System initialized. Student architecture: Compression');
}

// ==================== Обучение ====================
async function trainStep() {
    if (!inputTensor) return;
    
    // Baseline обучение (просто model.fit для простоты)
    await baselineModel.fit(inputTensor, inputTensor, {
        epochs: 1,
        verbose: 0
    });
    
    // Student обучение с кастомной функцией потерь
    tf.tidy(() => {
        const studentLossValue = optimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, { training: true });
            const loss = studentLoss(inputTensor, pred);
            return loss;
        }, true, [studentModel.getWeights()]);
        
        // Обновляем отображение
        step++;
        updateDisplays();
    });
    
    stepSpan.textContent = step;
}

// ==================== Визуализация ====================
function drawTensorToCanvas(tensor, canvas) {
    tf.tidy(() => {
        const squeezed = tensor.squeeze();
        const data = squeezed.dataSync();
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const cellSize = width / IMG_SIZE;
        
        ctx.clearRect(0, 0, width, height);
        
        for (let y = 0; y < IMG_SIZE; y++) {
            for (let x = 0; x < IMG_SIZE; x++) {
                const value = data[y * IMG_SIZE + x];
                const brightness = Math.floor(value * 255);
                ctx.fillStyle = `rgb(${brightness}, ${brightness}, ${brightness})`;
                ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
            }
        }
    });
}

async function updateDisplays() {
    if (!inputTensor || !baselineModel || !studentModel) return;
    
    tf.tidy(() => {
        // Предсказания
        const baselinePred = baselineModel.predict(inputTensor);
        const studentPred = studentModel.predict(inputTensor);
        
        // Потери
        const baselineLossVal = mse(inputTensor, baselinePred).dataSync()[0];
        const studentLossVal = studentLoss(inputTensor, studentPred).dataSync()[0];
        
        // Обновляем канвасы
        drawTensorToCanvas(baselinePred, baselineCanvas);
        drawTensorToCanvas(studentPred, studentCanvas);
        
        // Обновляем текст
        baselineLossSpan.textContent = baselineLossVal.toFixed(4);
        studentLossSpan.textContent = studentLossVal.toFixed(4);
    });
}

function updateStepDisplay() {
    stepSpan.textContent = step;
}

function log(message) {
    logArea.innerHTML += `<div>➡ ${message}</div>`;
    logArea.scrollTop = logArea.scrollHeight;
}

// ==================== Сброс ====================
async function reset() {
    isAutoTraining = false;
    autoTrainBtn.textContent = '▶ Auto Train';
    autoTrainBtn.classList.remove('stop');
    
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    
    // Пересоздаем модели
    tf.dispose([baselineModel, studentModel]);
    
    baselineModel = createBaselineModel();
    
    const selectedArch = Array.from(archRadios).find(r => r.checked).value;
    studentModel = createStudentModel(selectedArch);
    
    step = 0;
    updateStepDisplay();
    
    // Обновляем дисплей
    await updateDisplays();
    log(`🔄 Reset complete. Student architecture: ${selectedArch}`);
}

// ==================== Auto Train ====================
function startAutoTrain() {
    isAutoTraining = true;
    autoTrainBtn.textContent = '⏸ Stop';
    autoTrainBtn.classList.add('stop');
    
    async function trainLoop() {
        if (!isAutoTraining) return;
        
        for (let i = 0; i < 5; i++) { // 5 шагов за фрейм для скорости
            await trainStep();
        }
        
        animationFrame = requestAnimationFrame(trainLoop);
    }
    
    animationFrame = requestAnimationFrame(trainLoop);
}

function stopAutoTrain() {
    isAutoTraining = false;
    autoTrainBtn.textContent = '▶ Auto Train';
    autoTrainBtn.classList.remove('stop');
    
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
}

// ==================== Event Listeners ====================
trainOneBtn.addEventListener('click', async () => {
    await trainStep();
});

autoTrainBtn.addEventListener('click', () => {
    if (isAutoTraining) {
        stopAutoTrain();
    } else {
        startAutoTrain();
    }
});

resetBtn.addEventListener('click', reset);

archRadios.forEach(radio => {
    radio.addEventListener('change', async (e) => {
        if (e.target.checked) {
            // Пересоздаем студенческую модель с новой архитектурой
            const newArch = e.target.value;
            const oldWeights = studentModel.getWeights();
            
            studentModel = createStudentModel(newArch);
            
            // Копируем веса если возможно (для сохранения прогресса)
            try {
                const newWeights = studentModel.getWeights();
                for (let i = 0; i < Math.min(oldWeights.length, newWeights.length); i++) {
                    if (oldWeights[i].shape.join() === newWeights[i].shape.join()) {
                        newWeights[i].assign(oldWeights[i]);
                    }
                }
                studentModel.setWeights(newWeights);
            } catch (e) {
                console.log('Could not copy weights, starting fresh');
            }
            
            await updateDisplays();
            log(`🔄 Switched student architecture to: ${newArch}`);
        }
    });
});

// ==================== Запуск ====================
init();

// ==================== TODO-B: Функция для студентов (исправлена) ====================
// Внимание: Это ключевое место для эксперимента!
// Раскомментируйте и модифицируйте для создания градиента:
/*
function studentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        // База: MSE
        const mseLoss = mse(yTrue, yPred);
        
        // TODO: Добавьте smoothnessLoss и directionLoss с коэффициентами
        // Пример: 
        // const smoothLoss = smoothnessLoss(yPred);
        // const dirLoss = directionLoss(yPred);
        // 
        // return mseLoss.add(smoothLoss.mul(0.1)).add(dirLoss.mul(0.05));
        
        // Пока просто MSE
        return mseLoss;
    });
}
*/

// ==================== TODO-C: Сравнение ====================
// Для сравнения результатов используйте консоль браузера:
// console.log('Baseline vs Student:', {
//     baselineLoss: baselineLossSpan.textContent,
//     studentLoss: studentLossSpan.textContent,
//     step: step
// });