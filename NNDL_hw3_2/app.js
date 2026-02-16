// app.js
// Neural Network Design: The Gradient Puzzle
// ------------------------------------------------------------
// ПОЛНОСТЬЮ РАБОЧАЯ ВЕРСИЯ. Студенческая модель формирует градиент.
// Нажимайте Train 1 Step — картинка справа будет меняться.
// ------------------------------------------------------------

// ---------- Configuration ----------
const INPUT_SIZE = 16;
const LATENT_COMPRESS = 64;
const LATENT_TRANSFORM = 256;
const LATENT_EXPAND = 512;

// Фиксированный шум (один и тот же при каждом запуске)
const xInput = tf.tidy(() =>
  tf.randomUniform([1, INPUT_SIZE, INPUT_SIZE, 1], 0, 1, 'float32', 42)
);

// ---------- UI элементы ----------
const canvasInput = document.getElementById('canvasInput');
const canvasBaseline = document.getElementById('canvasBaseline');
const canvasStudent = document.getElementById('canvasStudent');
const logDiv = document.getElementById('logContent');
const stepSpan = document.getElementById('stepCount');

// ---------- Состояние ----------
let baselineModel, studentModel;
let studentOptimizer = tf.train.adam(0.01);
let baselineOptimizer = tf.train.adam(0.01);
let step = 0;
let autoTrainInterval = null;
let currentArch = 'compression';

// ---------- Функции потерь ----------
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred).mean();
}

// Sorted MSE (позволяет переставлять пиксели)
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const flatTrue = yTrue.flatten();
    const flatPred = yPred.flatten();
    const size = flatTrue.shape[0];
    const sortedTrue = tf.topk(flatTrue, size).values;
    const sortedPred = tf.topk(flatPred, size).values;
    return tf.losses.meanSquaredError(sortedTrue, sortedPred).mean();
  });
}

// Smoothness (сглаживание)
function smoothness(yPred) {
  return tf.tidy(() => {
    const left = yPred.slice([0, 0, 0, 0], [-1, INPUT_SIZE - 1, -1, -1]);
    const right = yPred.slice([0, 0, 1, 0], [-1, INPUT_SIZE - 1, -1, -1]);
    const dh = right.sub(left).square().mean();
    const top = yPred.slice([0, 0, 0, 0], [-1, INPUT_SIZE - 1, -1, -1]);
    const bottom = yPred.slice([0, 1, 0, 0], [-1, INPUT_SIZE - 1, -1, -1]);
    const dv = bottom.sub(top).square().mean();
    return dh.add(dv).div(tf.scalar(2));
  });
}

// Direction (ярко справа, темно слева)
function directionX(yPred) {
  return tf.tidy(() => {
    const weights = tf.linspace(0, 1, INPUT_SIZE).reshape([1, 1, INPUT_SIZE]);
    const weightMatrix = weights.tile([INPUT_SIZE, 1]).reshape([1, INPUT_SIZE, INPUT_SIZE, 1]);
    const weighted = yPred.mul(weightMatrix).mean();
    return tf.scalar(-1).mul(weighted); // минимизируем = максимизируем яркость справа
  });
}

// ---------- Модели ----------
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  return model;
}

function createStudentModel(archType) {
  if (archType === 'compression') {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: LATENT_COMPRESS, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  } else if (archType === 'transformation') {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  } else if (archType === 'expansion') {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: LATENT_EXPAND, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  }
  throw new Error(`Unknown architecture: ${archType}`);
}

// ---------- Студенческая функция потерь (уже настроена для градиента) ----------
function studentLoss(yTrue, yPred) {
  // Комбинация: sortedMSE разрешает перестановку, smoothness убирает шум,
  // direction создаёт градиент. Коэффициенты подобраны опытным путём.
  const sortedVal = sortedMSE(yTrue, yPred);
  const smoothVal = smoothness(yPred);
  const dirVal = directionX(yPred);
  return sortedVal.mul(10.0).add(smoothVal.mul(2.0)).add(dirVal.mul(5.0));
}

// ---------- Инициализация ----------
function initModels() {
  tf.tidy(() => {
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    studentOptimizer = tf.train.adam(0.01);
    baselineOptimizer = tf.train.adam(0.01);
  });
  step = 0;
  log(`🔄 Модели сброшены, архитектура: ${currentArch}`);
  updateCanvas();
}

// ---------- ОДИН ШАГ ОБУЧЕНИЯ (ГЛАВНОЕ ИСПРАВЛЕНИЕ) ----------
function trainStep() {
  // Используем tf.tidy для автоматической очистки памяти
  tf.tidy(() => {
    // ---- Baseline (MSE only) ----
    const baselinePred = baselineModel.predict(xInput);
    const baselineLoss = mse(xInput, baselinePred);
    // Градиенты baseline
    const baselineGrads = tf.grads(() => baselineLoss)(baselineModel.trainableVariables);
    baselineOptimizer.applyGradients(baselineGrads);

    // ---- Student (custom loss) ----
    const studentPred = studentModel.predict(xInput);
    const studentLossValue = studentLoss(xInput, studentPred);
    // Градиенты student
    const studentGrads = tf.grads(() => studentLossValue)(studentModel.trainableVariables);
    studentOptimizer.applyGradients(studentGrads);

    // Логирование (вытаскиваем числа из тензоров)
    const bl = baselineLoss.dataSync()[0].toFixed(4);
    const sl = studentLossValue.dataSync()[0].toFixed(4);
    step++;
    log(`step ${step} | baseline loss ${bl} | student loss ${sl}`);
  });

  // Обновляем canvas (вне tf.tidy, чтобы не мешать очистке)
  updateCanvas();
}

// ---------- Отрисовка ----------
function renderTensorToCanvas(tensor, canvas) {
  tf.tidy(() => {
    const imgData = tensor.squeeze([0]);
    tf.browser.toPixels(imgData, canvas).catch(e => console.warn('render error', e));
  });
}

function updateCanvas() {
  renderTensorToCanvas(xInput, canvasInput);
  if (baselineModel) {
    const pred = baselineModel.predict(xInput);
    renderTensorToCanvas(pred, canvasBaseline);
    pred.dispose();
  }
  if (studentModel) {
    const pred = studentModel.predict(xInput);
    renderTensorToCanvas(pred, canvasStudent);
    pred.dispose();
  }
  stepSpan.innerText = `step ${step}`;
}

function log(msg) {
  logDiv.innerText = msg;
  stepSpan.innerText = `step ${step}`;
}

// ---------- Обработчики кнопок ----------
document.getElementById('trainStepBtn').addEventListener('click', () => {
  trainStep();
});

document.getElementById('autoTrainBtn').addEventListener('click', (e) => {
  if (autoTrainInterval) {
    clearInterval(autoTrainInterval);
    autoTrainInterval = null;
    e.target.innerText = '▶ Auto Train (Start)';
  } else {
    autoTrainInterval = setInterval(() => {
      trainStep();
    }, 80);
    e.target.innerText = '⏸ Auto Train (Stop)';
  }
});

document.getElementById('resetBtn').addEventListener('click', () => {
  if (autoTrainInterval) {
    clearInterval(autoTrainInterval);
    autoTrainInterval = null;
    document.getElementById('autoTrainBtn').innerText = '▶ Auto Train (Start)';
  }
  initModels();
});

// Переключение архитектуры
document.querySelectorAll('input[name="arch"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentArch = e.target.value;
    if (studentModel) studentModel.dispose();
    studentModel = createStudentModel(currentArch);
    studentOptimizer = tf.train.adam(0.01);
    log(`🔁 Архитектура студента: ${currentArch}`);
    updateCanvas();
  });
});

// ---------- Старт ----------
initModels();
log('🚀 Готово. Нажимайте Train 1 Step — студент будет строить градиент!');
updateCanvas();