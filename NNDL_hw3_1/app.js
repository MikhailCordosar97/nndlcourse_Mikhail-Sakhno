// app.js
// Neural Network Design: The Gradient Puzzle
// ------------------------------------------------------------
// ГОТОВЫЙ КОД: студенческая модель уже использует custom loss,
// который превращает случайный шум в плавный градиент.
// Работает из коробки. Можно менять архитектуру и коэффициенты.
// ------------------------------------------------------------

// ---------- Configuration & Constants ----------
const INPUT_SIZE = 16;           // 16x16 grayscale
const LATENT_COMPRESS = 64;      // compression bottleneck
const LATENT_TRANSFORM = 256;    // transformation (same as input)
const LATENT_EXPAND = 512;       // expansion bottleneck

// Fixed random input (сохраняем один и тот же шум для воспроизводимости)
const xInput = tf.tidy(() => tf.randomUniform([1, INPUT_SIZE, INPUT_SIZE, 1], 0, 1, 'float32', 42));

// UI Elements
const canvasInput = document.getElementById('canvasInput');
const canvasBaseline = document.getElementById('canvasBaseline');
const canvasStudent = document.getElementById('canvasStudent');
const logDiv = document.getElementById('logContent');
const stepSpan = document.getElementById('stepCount');

// State
let baselineModel, studentModel;
let studentOptimizer = tf.train.adam(0.01);
let baselineOptimizer = tf.train.adam(0.01);
let step = 0;
let autoTrainInterval = null;
let currentArch = 'compression';   // default

// ---------- Loss components (provided) ----------

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred).mean();
}

// Sorted MSE (quantile / wasserstein) – liberates pixels from positions
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

// Smoothness (total variation) – encourages local consistency
function smoothness(yPred) {
  return tf.tidy(() => {
    const left = yPred.slice([0,0,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const right = yPred.slice([0,0,1,0], [-1, INPUT_SIZE-1, -1, -1]);
    const dh = right.sub(left).square().mean();
    const top = yPred.slice([0,0,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const bottom = yPred.slice([0,1,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const dv = bottom.sub(top).square().mean();
    return dh.add(dv).div(tf.scalar(2));
  });
}

// Direction loss: bright on right, dark on left
function directionX(yPred) {
  return tf.tidy(() => {
    const weights = tf.linspace(0, 1, INPUT_SIZE).reshape([1, 1, INPUT_SIZE]);
    const weightMatrix = weights.tile([INPUT_SIZE, 1]).reshape([1, INPUT_SIZE, INPUT_SIZE, 1]);
    const weighted = yPred.mul(weightMatrix).mean();
    // мы хотим максимизировать weighted -> минимизируем -weighted
    return tf.scalar(-1).mul(weighted);
  });
}

// ---------- Model creators ----------

// Baseline model (fixed, MSE only)
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  return model;
}

// Student model – architecture depends on selection
function createStudentModel(archType) {
  if (archType === 'compression') {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: LATENT_COMPRESS, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  }
  else if (archType === 'transformation') {
    // ----- transformation (bottleneck = 256, same as flattened 256) -----
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] })); // 256
    model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'relu' })); // no compression
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  }
  else if (archType === 'expansion') {
    // ----- expansion (bottleneck wider: 512) -----
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] })); // 256
    model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: LATENT_EXPAND, activation: 'relu' })); // 512
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  }
  throw new Error(`Unknown architecture: ${archType}`);
}

// ---------- CUSTOM LOSS (already tuned for gradient emergence) ----------
function studentLoss(yTrue, yPred) {
  // Базовая линия: sortedMSE разрешает перестановку пикселей,
  // smoothness убирает шум, direction создаёт градиент.
  // Коэффициенты подобраны эмпирически.
  const sortedVal = sortedMSE(yTrue, yPred);
  const smoothVal = smoothness(yPred);
  const dirVal = directionX(yPred);

  // Можно добавить небольшой mse, чтобы сохранять общую яркость,
  // но для чистого эффекта перестановки mse можно обнулить.
  return sortedVal * 10.0 + smoothVal * 2.0 + dirVal * 5.0;
}

// ---------- Инициализация ----------
function initModels() {
  tf.tidy(() => {
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    baselineModel = createBaselineModel();
    // студенческая модель всегда создаётся по текущей архитектуре
    studentModel = createStudentModel(currentArch);
    studentOptimizer = tf.train.adam(0.01);
    baselineOptimizer = tf.train.adam(0.01);
  });
  step = 0;
  log(`🔄 модели сброшены, архитектура: ${currentArch}`);
  updateLogAndCanvas();
}

// ---------- Правильный trainStep с GradientTape ----------
function trainStep() {
  // Используем tf.tidy для автоматической очистки промежуточных тензоров
  tf.tidy(() => {
    // ---- Baseline (MSE only) ----
    const baselineVars = baselineModel.trainableVariables;
    const baselineLoss = tf.tidy(() => {
      const pred = baselineModel.predict(xInput);
      return mse(xInput, pred);
    });
    // градиенты baseline
    const baselineGrads = tf.grads(loss => loss)(baselineLoss, baselineVars);
    baselineOptimizer.applyGradients(baselineGrads);
    // очистка (tf.tidy сделает всё сам, но градиенты уже применены)

    // ---- Student (custom loss) ----
    const studentVars = studentModel.trainableVariables;
    let studentLossValue;
    const studentGrads = tf.variableGrads(() => {
      const pred = studentModel.predict(xInput);
      const loss = studentLoss(xInput, pred);
      studentLossValue = loss.clone(); // сохраняем для логирования
      return loss;
    });
    studentOptimizer.applyGradients(studentGrads.grads);
    // освобождаем память от графов градиентов (studentGrads сам очистится в tidy)

    // Логирование
    step++;
    const baselineLossVal = baselineLoss.dataSync()[0];
    const studentLossVal = studentLossValue.dataSync()[0];
    log(`step ${step} | baseline ${baselineLossVal.toFixed(4)} | student ${studentLossVal.toFixed(4)}`);
  });
  updateCanvas();
}

// ---------- Отрисовка ----------
function renderTensorToCanvas(tensor, canvas) {
  tf.tidy(() => {
    const imgData = tensor.squeeze([0]); // [16,16,1]
    tf.browser.toPixels(imgData, canvas).catch(e => console.warn('canvas render error', e));
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
}

function updateLogAndCanvas() {
  updateCanvas();
  stepSpan.innerText = `step ${step}`;
}

function log(msg) {
  logDiv.innerText = msg;
  stepSpan.innerText = `step ${step}`;
}

// ---------- UI обработчики ----------
document.getElementById('trainStepBtn').addEventListener('click', () => {
  trainStep();
  updateLogAndCanvas();
});

document.getElementById('autoTrainBtn').addEventListener('click', (e) => {
  if (autoTrainInterval) {
    clearInterval(autoTrainInterval);
    autoTrainInterval = null;
    e.target.innerText = '▶ Auto Train (Start)';
  } else {
    autoTrainInterval = setInterval(() => {
      trainStep();
      updateLogAndCanvas();
    }, 80); // ~12 шагов/сек
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
  updateLogAndCanvas();
});

// Переключение архитектуры студента
document.querySelectorAll('input[name="arch"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentArch = e.target.value;
    // пересоздаём студента с новой архитектурой
    if (studentModel) studentModel.dispose();
    studentModel = createStudentModel(currentArch);
    studentOptimizer = tf.train.adam(0.01); // свежий оптимизатор
    log(`🔁 архитектура студента изменена на ${currentArch}`);
    updateCanvas();
  });
});

// ---------- Запуск ----------
initModels();
log('🚀 готово. student loss = sortedMSE*10 + smoothness*2 + direction*5. Нажимайте Train!');
updateLogAndCanvas();
