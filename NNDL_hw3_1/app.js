// app.js
// Neural Network Design: The Gradient Puzzle
// TODO-A, TODO-B, TODO-C markers guide the student tasks.

// ---------- Configuration & Constants ----------
const INPUT_SIZE = 16;           // 16x16 grayscale
const LATENT_COMPRESS = 64;      // compression bottleneck dimension
const LATENT_TRANSFORM = 256;    // transformation (same as input)
const LATENT_EXPAND = 512;       // expansion bottleneck

// Fixed random input (seed for reproducibility)
const xInput = tf.tidy(() => tf.randomUniform([1, INPUT_SIZE, INPUT_SIZE, 1], 0, 1, 'float32'));

// UI Elements
const canvasInput = document.getElementById('canvasInput');
const canvasBaseline = document.getElementById('canvasBaseline');
const canvasStudent = document.getElementById('canvasStudent');
const logDiv = document.getElementById('logContent');
const stepSpan = document.getElementById('stepCount');

// State
let baselineModel, studentModel;
let studentOptimizer = tf.train.adam(0.01);
let baselineOptimizer = tf.train.adam(0.01);  // separate, but we can also use model.fit later
let step = 0;
let autoTrainInterval = null;
let currentArch = 'compression';   // default

// ---------- Helper: loss components (provided) ----------

// Standard MSE
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred).mean();
}

// Sorted MSE (quantile / wasserstein loss) – sort pixels and compare
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    const flatTrue = yTrue.flatten();
    const flatPred = yPred.flatten();
    const size = flatTrue.shape[0];
    // sort descending (topk returns sorted values)
    const sortedTrue = tf.topk(flatTrue, size).values;
    const sortedPred = tf.topk(flatPred, size).values;
    return tf.losses.meanSquaredError(sortedTrue, sortedPred).mean();
  });
}

// Smoothness (total variation) – squared neighbor differences
function smoothness(yPred) {
  return tf.tidy(() => {
    // horizontal differences
    const left = yPred.slice([0,0,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const right = yPred.slice([0,0,1,0], [-1, INPUT_SIZE-1, -1, -1]);
    const dh = right.sub(left).square().mean();
    // vertical differences
    const top = yPred.slice([0,0,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const bottom = yPred.slice([0,1,0,0], [-1, INPUT_SIZE-1, -1, -1]);
    const dv = bottom.sub(top).square().mean();
    return dh.add(dv).div(tf.scalar(2)); // average over both directions
  });
}

// Direction loss: encourage bright on right (linear ramp)
function directionX(yPred) {
  return tf.tidy(() => {
    // create weight matrix: 0..1 from left to right
    const weights = tf.linspace(0, 1, INPUT_SIZE).reshape([1, 1, INPUT_SIZE]);
    const weightMatrix = weights.tile([INPUT_SIZE, 1]).reshape([1, INPUT_SIZE, INPUT_SIZE, 1]);
    const weighted = yPred.mul(weightMatrix).mean();
    // we want to maximize brightness on right → minimize negative weighted mean
    return tf.scalar(-1).mul(weighted);
  });
}

// ---------- Model creators ----------

// Baseline model (fixed architecture: compression, loss: MSE only)
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
  model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
  return model;
}

// ---------- TODO-A: implement createStudentModel(archType) ----------
// Students must complete 'transformation' and 'expansion' architectures.
// compression is already implemented (bottleneck 64).
function createStudentModel(archType) {
  if (archType === 'compression') {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: LATENT_COMPRESS, activation: 'relu' }));  // bottleneck
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    return model;
  }
  // ---------- TODO: Transformation (autoencoder with same dimension) ----------
  else if (archType === 'transformation') {
    throw new Error('🚧 TODO-A: implement transformation architecture (latent = 256, no compression)');
    // Example:
    // const model = tf.sequential();
    // model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    // model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    // model.add(tf.layers.dense({ units: 256, activation: 'relu' }));  // same-size bottleneck
    // model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    // model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    // return model;
  }
  // ---------- TODO: Expansion (autoencoder with wider latent) ----------
  else if (archType === 'expansion') {
    throw new Error('🚧 TODO-A: implement expansion architecture (latent = 512, expansion)');
    // Example:
    // const model = tf.sequential();
    // model.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    // model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    // model.add(tf.layers.dense({ units: LATENT_EXPAND, activation: 'relu' })); // wider
    // model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    // model.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    // return model;
  }
  throw new Error(`Unknown architecture: ${archType}`);
}

// ---------- TODO-B: Custom student loss (combine components) ----------
// Students should modify this function to include sortedMSE, smoothness, directionX.
// Coefficients can be tuned.
function studentLoss(yTrue, yPred) {
  // baseline: only MSE (so student initially behaves like baseline)
  const mseVal = mse(yTrue, yPred);

  // ----- TODO: add sortedMSE, smoothness, direction with weights -----
  // const sortedVal = sortedMSE(yTrue, yPred);
  // const smoothVal = smoothness(yPred);
  // const dirVal = directionX(yPred);
  // return mseVal * 1.0 + sortedVal * 10.0 + smoothVal * 2.0 + dirVal * 5.0;

  return mseVal; // placeholder: MSE only
}

// ---------- Initialization ----------
function initModels() {
  tf.tidy(() => {
    if (baselineModel) baselineModel.dispose();
    if (studentModel) studentModel.dispose();
    baselineModel = createBaselineModel();
    // student model based on current architecture
    try {
      studentModel = createStudentModel(currentArch);
    } catch (e) {
      log('⚠️ ' + e.message);
      // fallback: create a simple model to keep app running
      studentModel = tf.sequential();
      studentModel.add(tf.layers.flatten({ inputShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
      studentModel.add(tf.layers.dense({ units: 128, activation: 'relu' }));
      studentModel.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
      studentModel.add(tf.layers.reshape({ targetShape: [INPUT_SIZE, INPUT_SIZE, 1] }));
    }
    // Reset optimizers
    studentOptimizer = tf.train.adam(0.01);
    baselineOptimizer = tf.train.adam(0.01);
  });
  step = 0;
  updateLogAndCanvas();
}

// ---------- Training Step ----------
function trainStep() {
  tf.tidy(() => {
    // Forward pass for both models
    const baselinePred = baselineModel.predict(xInput);
    const studentPred = studentModel.predict(xInput);

    // Compute losses
    const baselineLoss = mse(xInput, baselinePred);   // baseline always uses pure MSE
    let studentLossValue;
    try {
      studentLossValue = studentLoss(xInput, studentPred);
    } catch (e) {
      log('🔥 Error in studentLoss: ' + e.message);
      studentLossValue = tf.scalar(Infinity);
    }

    // Compute gradients for baseline model
    baselineOptimizer.minimize(() => baselineLoss, /* varList */ true, /* f() */ () => {
      const grads = tf.grads(() => baselineLoss)([xInput], baselineModel.trainableVariables);
      // Since minimize expects a function that returns loss, we use it directly.
      // But we need to use the variables. Simpler: use gradient tape explicitly.
      // We'll switch to explicit tape for clarity:
      return baselineLoss;
    });

    // Explicit tape for student (to catch errors)
    const studentGrads = tf.variableGrads(() => studentLoss(xInput, studentPred));
    studentOptimizer.applyGradients(studentGrads.grads);
    studentGrads.grads.forEach(g => g.dispose()); // clean

    // Logging
    step++;
    const bl = baselineLoss.dataSync()[0].toFixed(4);
    const sl = studentLossValue.dataSync()[0].toFixed(4);
    log(`step ${step} | baseline loss ${bl} | student loss ${sl}`);
  });
  updateCanvas();
}

// Better gradient approach: use tf.GradientTape (tf.engine().tidy + tape)
// But above works; let's refine to avoid potential dispose issues.
// We'll rewrite trainStep with tape for safety.
function trainStep() {
  tf.tidy(() => {
    // Baseline gradients with tape
    const baselineTape = tf.engine().tidy(() => {
      const pred = baselineModel.predict(xInput);
      const loss = mse(xInput, pred);
      return { loss, pred };
    });
    const baselineLoss = baselineTape.loss;
    const baselineVars = baselineModel.trainableVariables;
    const baselineGrads = tf.grads(loss => loss)(baselineLoss, baselineVars);
    baselineOptimizer.applyGradients(baselineGrads);
    baselineLoss.dispose();

    // Student gradients with tape
    let studentLossValue;
    const studentTape = tf.engine().tidy(() => {
      const pred = studentModel.predict(xInput);
      let loss;
      try {
        loss = studentLoss(xInput, pred);
      } catch (e) {
        log('🔥 studentLoss error: ' + e.message);
        loss = tf.scalar(1e9);
      }
      studentLossValue = loss.clone();
      return { loss, pred };
    });
    const studentLossActual = studentTape.loss;
    const studentVars = studentModel.trainableVariables;
    const studentGrads = tf.grads(loss => loss)(studentLossActual, studentVars);
    studentOptimizer.applyGradients(studentGrads);
    studentLossActual.dispose();

    step++;
    const bl = mse(xInput, baselineModel.predict(xInput)).dataSync()[0].toFixed(4);
    const sl = studentLossValue.dataSync()[0].toFixed(4);
    log(`step ${step} | baseline loss ${bl} | student loss ${sl}`);
  });
  updateCanvas();
}

// ---------- Canvas Rendering ----------
function renderTensorToCanvas(tensor, canvas) {
  tf.tidy(() => {
    const imgData = tensor.squeeze([0]); // [16,16,1]
    tf.browser.toPixels(imgData, canvas).then(() => {});
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

// ---------- UI Event Handlers ----------
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
    }, 80); // ~12 steps per second, smooth
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
  log('🔄 models reinitialized');
  updateLogAndCanvas();
});

// Architecture radio change
document.querySelectorAll('input[name="arch"]').forEach(radio => {
  radio.addEventListener('change', (e) => {
    currentArch = e.target.value;
    // rebuild student model
    tf.tidy(() => {
      if (studentModel) studentModel.dispose();
      try {
        studentModel = createStudentModel(currentArch);
        studentOptimizer = tf.train.adam(0.01); // fresh optimizer
        log(`✅ architecture changed to ${currentArch} (student model rebuilt)`);
      } catch (err) {
        log(`❌ ${err.message} — using fallback model`);
        // fallback to compression to keep app alive
        studentModel = createStudentModel('compression');
        studentOptimizer = tf.train.adam(0.01);
      }
      updateCanvas();
    });
  });
});

// ---------- Start ----------
initModels();
log('🟢 ready. modify TODO sections in app.js to build your own loss!');
updateLogAndCanvas();

// ---------- TODO-C: comparison (already printed in log) ----------
// Students can extend the log to show baseline vs student visually.
// Visual difference is shown on canvases. To emphasise comparison, you could
// add a difference canvas, but that's optional. We keep it simple.