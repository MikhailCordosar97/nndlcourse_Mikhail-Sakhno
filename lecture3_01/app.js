// app.js - The Gradient Puzzle: Distribution MSE + Shape Constraints
// Using histogram matching instead of sorting for gradient compatibility

class GradientPuzzle {
    constructor() {
        this.stepCount = 0;
        this.isTraining = false;
        this.animationId = null;
        this.xInput = null;
        this.baselineModel = null;
        this.studentModel = null;
        this.optimizer = tf.train.adam(0.01);
        
        this.smoothnessWeight = 0.0;
        this.directionWeight = 0.0;
        
        // For histogram matching
        this.numBins = 32;
        this.binEdges = null;
        
        this.init();
    }

    async init() {
        try {
            await tf.ready();
            this.log(`TensorFlow.js backend: ${tf.getBackend()}`);
            
            // Generate fixed 16x16 noise input
            this.xInput = tf.tidy(() => {
                return tf.randomNormal([1, 16, 16, 1]).clipByValue(-1, 1);
            });
            
            // Precompute bin edges for histogram
            this.binEdges = tf.linspace(-1, 1, this.numBins + 1);
            
            // Create identical models
            this.baselineModel = this.createModel();
            this.studentModel = this.createModel();
            
            this.setupUI();
            this.renderAll();
            this.log('Ready. Adjust sliders and click "Train 1 Step".');
            
        } catch (error) {
            this.log(`Initialization error: ${error.message}`, true);
        }
    }

    createModel() {
        // Simple autoencoder architecture
        const model = tf.sequential();
        
        // Encoder
        model.add(tf.layers.conv2d({
            filters: 8,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu',
            inputShape: [16, 16, 1]
        }));
        model.add(tf.layers.conv2d({
            filters: 4,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu'
        }));
        
        // Decoder
        model.add(tf.layers.conv2dTranspose({
            filters: 8,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'relu'
        }));
        model.add(tf.layers.conv2dTranspose({
            filters: 1,
            kernelSize: 3,
            strides: 2,
            padding: 'same',
            activation: 'tanh'
        }));
        
        return model;
    }

    // ============= LOSS FUNCTIONS =============
    
    pixelMSE(yTrue, yPred) {
        // Standard pixel-wise MSE: FIXES POSITIONS
        return tf.tidy(() => {
            return tf.losses.meanSquaredError(yTrue, yPred);
        });
    }

    computeHistogram(tensor) {
        // Compute histogram with differentiable binning using soft assignment
        return tf.tidy(() => {
            const flat = tf.reshape(tensor, [-1]); // Flatten to [256]
            
            // Compute distances to bin edges
            const distances = tf.sub(flat.expandDims(1), this.binEdges.expandDims(0));
            
            // Soft assignment: use sigmoid to create smooth bin membership
            const sigma = 0.1; // Controls smoothness of bin assignment
            const leftEdges = tf.sigmoid(tf.neg(distances.slice([0, 0], [-1, this.numBins])).div(sigma));
            const rightEdges = tf.sigmoid(tf.neg(distances.slice([0, 1], [-1, this.numBins])).div(sigma));
            
            // Bin membership is difference between left and right edge assignments
            const binMembership = leftEdges.sub(rightEdges);
            
            // Sum across pixels and normalize to get probability distribution
            const histogram = binMembership.sum(0).div(flat.shape[0]);
            
            return histogram;
        });
    }

    distributionMSE(yTrue, yPred) {
        // Match histogram/distribution instead of exact positions
        return tf.tidy(() => {
            const histTrue = this.computeHistogram(yTrue);
            const histPred = this.computeHistogram(yPred);
            
            // KL divergence between distributions (more stable for training)
            const epsilon = 1e-7;
            const histTrueSafe = histTrue.add(epsilon);
            const histPredSafe = histPred.add(epsilon);
            
            // KL(P||Q) = sum P * log(P/Q)
            const kl = histTrueSafe.mul(histTrueSafe.div(histPredSafe).log()).mean();
            
            return kl;
        });
    }

    smoothnessLoss(yPred) {
        // Total variation: encourages local smoothness
        return tf.tidy(() => {
            // Horizontal differences
            const diffX = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
                          .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
            
            // Vertical differences
            const diffY = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1])
                          .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
            
            return diffX.square().mean().add(diffY.square().mean());
        });
    }

    directionLoss(yPred) {
        // Encourages left-dark, right-bright gradient
        return tf.tidy(() => {
            const [batch, height, width, channels] = yPred.shape;
            
            // Create x-coordinate grid from -1 (left) to +1 (right)
            const xCoords = tf.linspace(-1, 1, width)
                .reshape([1, 1, width, 1])
                .tile([batch, height, 1, channels]);
            
            // Ideal output: linear gradient matching x-coordinate
            const ideal = xCoords;
            
            // MSE between current output and ideal gradient
            return tf.losses.meanSquaredError(ideal, yPred);
        });
    }

    studentLoss(yTrue, yPred) {
        // Combined loss: Distribution matching + Shape constraints
        return tf.tidy(() => {
            // Base loss: distribution matching (allows pixel movement)
            const distLoss = this.distributionMSE(yTrue, yPred);
            
            // Shape constraints (tunable via UI sliders)
            const smoothLoss = this.smoothnessLoss(yPred).mul(this.smoothnessWeight);
            const dirLoss = this.directionLoss(yPred).mul(this.directionWeight);
            
            // Combined loss
            const totalLoss = distLoss.add(smoothLoss).add(dirLoss);
            
            return totalLoss;
        });
    }

    // ============= TRAINING =============
    
    async trainStep() {
        if (!this.xInput || !this.baselineModel || !this.studentModel) return;
        
        try {
            const losses = await tf.tidy(() => {
                // Forward pass
                const baselinePred = this.baselineModel.predict(this.xInput);
                const studentPred = this.studentModel.predict(this.xInput);
                
                // Compute losses
                const baselineLoss = this.pixelMSE(this.xInput, baselinePred);
                const studentLoss = this.studentLoss(this.xInput, studentPred);
                
                // Baseline gradients (pixel MSE)
                const baselineVars = this.baselineModel.trainableWeights;
                const baselineGrads = tf.grad(
                    () => this.pixelMSE(this.xInput, this.baselineModel.predict(this.xInput))
                );
                this.optimizer.applyGradients(
                    baselineGrads(baselineVars).map((grad, i) => ({
                        tensor: grad,
                        variable: baselineVars[i]
                    }))
                );
                
                // Student gradients (distribution matching + constraints)
                const studentVars = this.studentModel.trainableWeights;
                const studentGrads = tf.grad(
                    () => this.studentLoss(this.xInput, this.studentModel.predict(this.xInput))
                );
                this.optimizer.applyGradients(
                    studentGrads(studentVars).map((grad, i) => ({
                        tensor: grad,
                        variable: studentVars[i]
                    }))
                );
                
                // Render
                this.renderOutput(baselinePred, 'baselineCanvas');
                this.renderOutput(studentPred, 'studentCanvas');
                this.renderHistograms();
                
                return {
                    baseline: baselineLoss,
                    student: studentLoss
                };
            });
            
            this.stepCount++;
            this.log(`Step ${this.stepCount}: ` +
                     `Baseline (Pixel MSE) = ${losses.baseline.dataSync()[0].toFixed(4)} | ` +
                     `Student (Distribution+Constraints) = ${losses.student.dataSync()[0].toFixed(4)}`);
            
        } catch (error) {
            this.log(`Training error: ${error.message}`, true);
            this.stopTraining();
        }
        
        tf.engine().endScope();
    }

    // ============= UI & RENDERING =============
    
    setupUI() {
        // Control buttons
        document.getElementById('trainStepBtn').addEventListener('click', () => {
            this.trainStep();
        });
        
        const autoBtn = document.getElementById('autoTrainBtn');
        autoBtn.addEventListener('click', () => {
            if (this.isTraining) {
                this.stopTraining();
                autoBtn.textContent = 'Auto Train (Start)';
                autoBtn.classList.remove('active');
            } else {
                this.startTraining();
                autoBtn.textContent = 'Auto Train (Stop)';
                autoBtn.classList.add('active');
            }
        });
        
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetWeights();
        });
        
        // Loss parameter sliders
        const smoothSlider = document.getElementById('smoothnessSlider');
        const smoothValue = document.getElementById('smoothnessValue');
        smoothSlider.addEventListener('input', (e) => {
            this.smoothnessWeight = parseFloat(e.target.value);
            smoothValue.textContent = this.smoothnessWeight.toFixed(2);
            this.log(`Smoothness weight: ${this.smoothnessWeight.toFixed(2)}`);
        });
        
        const dirSlider = document.getElementById('directionSlider');
        const dirValue = document.getElementById('directionValue');
        dirSlider.addEventListener('input', (e) => {
            this.directionWeight = parseFloat(e.target.value);
            dirValue.textContent = this.directionWeight.toFixed(2);
            this.log(`Direction weight: ${this.directionWeight.toFixed(2)}`);
        });
    }

    startTraining() {
        if (this.isTraining) return;
        this.isTraining = true;
        
        const trainLoop = () => {
            if (!this.isTraining) return;
            
            // Train 2 steps per frame
            for (let i = 0; i < 2 && this.isTraining; i++) {
                this.trainStep();
            }
            
            if (this.isTraining) {
                this.animationId = requestAnimationFrame(trainLoop);
            }
        };
        
        this.animationId = requestAnimationFrame(trainLoop);
    }

    stopTraining() {
        this.isTraining = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    resetWeights() {
        this.stopTraining();
        
        // Reinitialize models
        this.baselineModel = this.createModel();
        this.studentModel = this.createModel();
        
        this.stepCount = 0;
        this.renderAll();
        this.log('Weights reset. Ready for training.');
    }

    renderAll() {
        this.renderInput();
        
        if (this.baselineModel && this.studentModel) {
            tf.tidy(() => {
                const baselinePred = this.baselineModel.predict(this.xInput);
                const studentPred = this.studentModel.predict(this.xInput);
                
                this.renderOutput(this.xInput, 'inputCanvas');
                this.renderOutput(baselinePred, 'baselineCanvas');
                this.renderOutput(studentPred, 'studentCanvas');
                this.renderHistograms();
            });
        }
    }

    renderInput() {
        this.renderOutput(this.xInput, 'inputCanvas');
    }

    renderOutput(tensor, canvasId) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        tf.tidy(() => {
            // Scale 16x16 to 256x256
            const scaled = tf.image.resizeBilinear(tensor, [256, 256]);
            const data = scaled.dataSync();
            
            const imageData = ctx.createImageData(256, 256);
            for (let i = 0; i < data.length; i++) {
                const val = Math.floor((data[i] + 1) * 127.5);
                imageData.data[i * 4] = val;
                imageData.data[i * 4 + 1] = val;
                imageData.data[i * 4 + 2] = val;
                imageData.data[i * 4 + 3] = 255;
            }
            
            ctx.putImageData(imageData, 0, 0);
        });
    }

    renderHistograms() {
        tf.tidy(() => {
            const baselinePred = this.baselineModel.predict(this.xInput);
            const studentPred = this.studentModel.predict(this.xInput);
            
            this.renderHistogram(this.xInput, 'inputHistogram', '#60a5fa');
            this.renderHistogram(baselinePred, 'baselineHistogram', '#10b981');
            this.renderHistogram(studentPred, 'studentHistogram', '#f59e0b');
        });
    }

    renderHistogram(tensor, canvasId, color) {
        const canvas = document.getElementById(canvasId);
        const ctx = canvas.getContext('2d');
        
        // Clear
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Compute histogram (non-differentiable version for display)
        const data = tensor.dataSync();
        const bins = new Array(this.numBins).fill(0);
        
        data.forEach(value => {
            // Map value from [-1, 1] to bin index [0, numBins-1]
            const binIndex = Math.min(
                Math.floor((value + 1) * this.numBins / 2),
                this.numBins - 1
            );
            if (binIndex >= 0 && binIndex < this.numBins) {
                bins[binIndex]++;
            }
        });
        
        // Normalize
        const max = Math.max(...bins);
        if (max === 0) return;
        
        // Draw bars
        ctx.fillStyle = color;
        const barWidth = canvas.width / this.numBins;
        
        for (let i = 0; i < this.numBins; i++) {
            const height = (bins[i] / max) * canvas.height;
            ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 1, height);
        }
        
        // Draw median line
        const sortedData = [...data].sort((a, b) => a - b);
        const median = sortedData[Math.floor(sortedData.length / 2)];
        const medianX = ((median + 1) / 2) * canvas.width;
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(medianX, 0);
        ctx.lineTo(medianX, canvas.height);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    log(message, isError = false) {
        const logElement = document.getElementById('log');
        const entry = document.createElement('div');
        entry.className = `log-entry ${isError ? 'error' : ''}`;
        
        const timestamp = new Date().toLocaleTimeString();
        const htmlMessage = message
            .replace(/Step (\d+)/g, '<span class="step-count">Step $1</span>')
            .replace(/Baseline.*?=\s*([\d.]+)/g, '<span class="loss-baseline">Baseline = $1</span>')
            .replace(/Student.*?=\s*([\d.]+)/g, '<span class="loss-student">Student = $1</span>');
        
        entry.innerHTML = `[${timestamp}] ${htmlMessage}`;
        logElement.appendChild(entry);
        logElement.scrollTop = logElement.scrollHeight;
        
        // Keep only last 30 entries
        const entries = logElement.querySelectorAll('.log-entry');
        if (entries.length > 30) entries[0].remove();
        
        if (isError) console.error(message);
        else console.log(message);
    }
}

// Initialize app
let app;
window.addEventListener('DOMContentLoaded', () => {
    app = new GradientPuzzle();
});

// Cleanup
window.addEventListener('beforeunload', () => {
    if (app) {
        app.stopTraining();
        tf.dispose([app.xInput, app.baselineModel, app.studentModel]);
    }
});