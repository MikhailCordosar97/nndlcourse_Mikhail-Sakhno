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

// Smoothness Loss (Total Variation)
function smoothnessLoss(yPred) {
    return tf.tidy(() => {
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
    });
}

// Direction Loss - поощряет градиент слева направо
function directionLoss(yPred) {
    return tf.tidy(() => {
        const [batch, height, width, channels] = yPred.shape;
        
        // Создаем маску: значения от 0 до 1 слева направо
        const range = tf.linspace(0, 1, width);
        const mask2d = tf.tile(range.reshape([1, width]), [height, 1]);
        const mask = mask2d.reshape([1, height, width, 1]);
        
        // Поощряем соответствие маске (чем больше значение * маска, тем лучше)
        // Используем отрицательный знак для минимизации
        return tf.neg(tf.mean(tf.mul(yPred, mask)));
    });
}

// ==================== Студенческая функция потерь (С РЕШЕНИЕМ) ====================
function computeStudentLoss(yTrue, yPred) {
    return tf.tidy(() => {
        // Базовая MSE потеря
        const mseLoss = mse(yTrue, yPred);
        
        // Добавляем smoothness для плавности
        const smoothLoss = smoothnessLoss(yPred);
        
        // Добавляем direction для градиента
        const dirLoss = directionLoss(yPred);
        
        // Комбинируем с коэффициентами
        // Коэффициенты подобраны для хорошего градиента
        const total = tf.add(
            tf.add(
                mseLoss,
                tf.mul(smoothLoss, 0.2)
            ),
            tf.mul(dirLoss, 0.1)
        );
        
        return total;
    });
}

// ==================== Создание моделей ====================
function createBaselineModel() {
    const model = tf.sequential();
    
    model.add(tf.layers.conv2d({
        inputShape: [IMG_SIZE, IMG_SIZE, 1],
        filters: 16,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
    
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
    
    return model;
}

function createStudentModel(archType) {
    const model = tf.sequential();
    
    switch(archType) {
        case 'compression':
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
                filters: 16,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.maxPooling2d({ poolSize: 2, padding: 'same' }));
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
            break;
            
        case 'transformation':
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
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
                filters: 16,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            break;
            
        case 'expansion':
            model.add(tf.layers.conv2d({
                inputShape: [IMG_SIZE, IMG_SIZE, 1],
                filters: 32,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            model.add(tf.layers.conv2d({
                filters: 64,
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
                filters: 16,
                kernelSize: 3,
                padding: 'same',
                activation: 'relu'
            }));
            break;
    }
    
    // Выходной слой (общий для всех архитектур)
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
    
    // Создаем фиксированный шум
    inputTensor = tf.tidy(() => {
        return tf.randomUniform([1, IMG_SIZE, IMG_SIZE, 1], 0, 1);
    });
    
    // Рисуем вход
    await drawTensorToCanvas(inputTensor, inputCanvas);
    
    // Создаем модели
    baselineModel = createBaselineModel();
    studentModel = createStudentModel('compression');
    
    // Оптимизатор для кастомного обучения
    optimizer = tf.train.adam(0.01);
    
    step = 0;
    stepSpan.textContent = step;
    
    // Первое обновление дисплея
    await updateDisplays();
    
    log('✅ Ready! Press "Train 1 Step" to begin.');
}

// ==================== Обучение ====================
async function trainStep() {
    if (!inputTensor || !baselineModel || !studentModel) {
        log('❌ Models not initialized');
        return;
    }
    
    try {
        // Baseline обучение
        await baselineModel.fit(inputTensor, inputTensor, {
            epochs: 1,
            verbose: 0
        });
        
        // Student обучение с кастомной функцией потерь
        optimizer.minimize(() => {
            const pred = studentModel.apply(inputTensor, { training: true });
            const loss = computeStudentLoss(inputTensor, pred);
            return loss;
        });
        
        step++;
        stepSpan.textContent = step;
        
        // Обновляем дисплей
        await updateDisplays();
        
        if (step % 10 === 0) {
            log(`Step ${step} completed`);
        }
    } catch (error) {
        log(`❌ Error: ${error.message}`);
        console.error(error);
    }
}

// ==================== Визуализация ====================
async function drawTensorToCanvas(tensor, canvas) {
    return new Promise((resolve) => {
        tf.tidy(() => {
            const data = tensor.squeeze().dataSync();
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;
            const cellSize = width / IMG_SIZE;
            
            // Создаем ImageData для более быстрой отрисовки
            const imageData = ctx.createImageData(width, height);
            
            for (let y = 0; y < IMG_SIZE; y++) {
                for (let x = 0; x < IMG_SIZE; x++) {
                    const value = data[y * IMG_SIZE + x];
                    const brightness = Math.floor(value * 255);
                    
                    // Заполняем пиксель (увеличиваем для четкости)
                    for (let dy = 0; dy < cellSize; dy++) {
                        for (let dx = 0; dx < cellSize; dx++) {
                            const px = x * cellSize + dx;
                            const py = y * cellSize + dy;
                            const index = (py * width + px) * 4;
                            
                            imageData.data[index] = brightness;     // R
                            imageData.data[index + 1] = brightness; // G
                            imageData.data[index + 2] = brightness; // B
                            imageData.data[index + 3] = 255;        // A
                        }
                    }
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
            resolve();
        });
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
        const studentLossVal = computeStudentLoss(inputTensor, studentPred).dataSync()[0];
        
        // Обновляем канвасы (асинхронно)
        drawTensorToCanvas(baselinePred, baselineCanvas);
        drawTensorToCanvas(studentPred, studentCanvas);
        
        // Обновляем текст
        baselineLossSpan.textContent = baselineLossVal.toFixed(6);
        studentLossSpan.textContent = studentLossVal.toFixed(6);
    });
}

function log(message) {
    const timestamp = new Date().toLocaleTimeString();
    logArea.innerHTML += `<div>[${timestamp}] ${message}</div>`;
    logArea.scrollTop = logArea.scrollHeight;
}

// ==================== Сброс ====================
async function reset() {
    log('Resetting...');
    
    // Останавливаем автообучение
    if (isAutoTraining) {
        stopAutoTrain();
    }
    
    // Очищаем память
    if (baselineModel) tf.dispose(baselineModel);
    if (studentModel) tf.dispose(studentModel);
    
    // Создаем новые модели
    baselineModel = createBaselineModel();
    
    const selectedArch = Array.from(archRadios).find(r => r.checked).value;
    studentModel = createStudentModel(selectedArch);
    
    // Пересоздаем оптимизатор
    optimizer = tf.train.adam(0.01);
    
    step = 0;
    stepSpan.textContent = step;
    
    // Обновляем дисплей
    await updateDisplays();
    
    log(`✅ Reset complete. Student architecture: ${selectedArch}`);
}

// ==================== Auto Train ====================
function startAutoTrain() {
    isAutoTraining = true;
    autoTrainBtn.textContent = '⏸ Stop';
    autoTrainBtn.classList.add('stop');
    
    async function trainLoop() {
        if (!isAutoTraining) return;
        
        await trainStep();
        
        // Задержка для видимости процесса
        setTimeout(() => {
            animationFrame = requestAnimationFrame(trainLoop);
        }, 50);
    }
    
    animationFrame = requestAnimationFrame(trainLoop);
    log('▶ Auto training started');
}

function stopAutoTrain() {
    isAutoTraining = false;
    autoTrainBtn.textContent = '▶ Auto Train';
    autoTrainBtn.classList.remove('stop');
    
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
        animationFrame = null;
    }
    
    log('⏸ Auto training stopped');
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
            log(`Switching to ${e.target.value} architecture...`);
            
            // Сохраняем старые веса если возможно
            const oldWeights = studentModel ? studentModel.getWeights() : null;
            
            // Создаем новую модель
            studentModel = createStudentModel(e.target.value);
            
            // Пытаемся скопировать веса из старой модели
            if (oldWeights) {
                try {
                    const newWeights = studentModel.getWeights();
                    for (let i = 0; i < Math.min(oldWeights.length, newWeights.length); i++) {
                        if (oldWeights[i].shape.join() === newWeights[i].shape.join()) {
                            newWeights[i].assign(oldWeights[i]);
                        }
                    }
                    studentModel.setWeights(newWeights);
                    log('✅ Preserved some weights');
                } catch (e) {
                    log('ℹ Started with fresh weights');
                }
            }
            
            await updateDisplays();
            log(`✅ Switched to ${e.target.value}`);
        }
    });
});

// ==================== Запуск ====================
// Ждем загрузки TensorFlow.js
tf.ready().then(() => {
    log('TensorFlow.js loaded');
    init();
});

// Очистка памяти при закрытии
window.addEventListener('beforeunload', () => {
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }
    tf.dispose([inputTensor, baselineModel, studentModel]);
});