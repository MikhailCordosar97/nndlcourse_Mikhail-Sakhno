// Import Transformers.js pipeline
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// DOM elements
const reviewDisplay = document.getElementById('reviewDisplay');
const analyzeButton = document.getElementById('analyzeButton');
const sentimentIcon = document.getElementById('sentimentIcon');
const sentimentLabel = document.getElementById('sentimentLabel');
const confidence = document.getElementById('confidence');
const resultContainer = document.getElementById('resultContainer');
const statusText = document.getElementById('statusText');
const errorPanel = document.getElementById('errorPanel');
const errorText = document.getElementById('errorText');
const reviewStats = document.getElementById('reviewStats');
const modelStatus = document.getElementById('modelStatus');
const tsvFileInput = document.getElementById('tsvFile');
const useSamplesButton = document.getElementById('useSamples');

// Application state
let reviews = [];
let sentimentPipeline = null;
let isModelReady = false;
let isReviewsLoaded = false;

// Sample reviews for testing
const SAMPLE_REVIEWS = [
    "This product is absolutely amazing! The quality exceeded my expectations and it was worth every penny.",
    "Terrible experience. The product broke after just 2 days of use and customer service was unhelpful.",
    "It's okay for the price, but nothing special. Does the job but I expected better quality.",
    "I love this product! It has completely changed how I approach my daily routine. Highly recommended!",
    "The worst purchase I've ever made. Save your money and look elsewhere.",
    "Decent product with some flaws. The design could be improved but overall it works fine.",
    "Excellent value for money. The features are robust and the performance is outstanding.",
    "Very disappointed with this purchase. The product arrived damaged and the replacement was just as bad.",
    "Good quality and reasonable price. I would buy this again.",
    "Not sure how I feel about this. Some aspects are great but others are frustrating."
];

/**
 * Update status message in UI
 */
function updateStatus(message, isError = false) {
    statusText.innerHTML = message;
    
    if (isError) {
        console.error(message);
    } else {
        console.log(message);
    }
}

/**
 * Show error message in UI
 */
function showError(message) {
    errorText.textContent = message;
    errorPanel.style.display = 'block';
    console.error(message);
}

/**
 * Hide error message in UI
 */
function hideError() {
    errorPanel.style.display = 'none';
}

/**
 * Parse TSV content and extract reviews
 */
function parseTSVContent(tsvContent) {
    try {
        // Parse TSV using Papa Parse
        const result = Papa.parse(tsvContent, {
            header: true,
            delimiter: '\t',
            skipEmptyLines: true
        });
        
        if (result.errors && result.errors.length > 0) {
            console.warn('TSV parsing warnings:', result.errors);
        }
        
        // Extract review texts from the 'text' column
        reviews = result.data
            .map(row => row.text)
            .filter(text => text && typeof text === 'string' && text.trim().length > 0)
            .map(text => text.trim());
        
        return reviews.length > 0;
        
    } catch (error) {
        console.error('TSV parsing error:', error);
        return false;
    }
}

/**
 * Load reviews from a file
 */
async function loadReviewsFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        
        reader.onload = function(event) {
            try {
                const fileContent = event.target.result;
                const success = parseTSVContent(fileContent);
                
                if (success) {
                    resolve(reviews.length);
                } else {
                    reject(new Error('No valid reviews found in the file. Please check the format.'));
                }
            } catch (error) {
                reject(error);
            }
        };
        
        reader.onerror = function() {
            reject(new Error('Failed to read the file.'));
        };
        
        reader.readAsText(file);
    });
}

/**
 * Load sample reviews
 */
function loadSampleReviews() {
    reviews = [...SAMPLE_REVIEWS];
    return reviews.length;
}

/**
 * Update UI after reviews are loaded
 */
function updateReviewsUI() {
    isReviewsLoaded = reviews.length > 0;
    reviewStats.textContent = `Reviews loaded: ${reviews.length}`;
    
    if (isReviewsLoaded) {
        updateStatus(`Successfully loaded ${reviews.length} reviews.`);
        analyzeButton.disabled = !isModelReady;
    } else {
        updateStatus('No reviews loaded.', true);
        analyzeButton.disabled = true;
    }
}

/**
 * Initialize the sentiment analysis model
 */
async function initializeModel() {
    try {
        updateStatus('Loading sentiment analysis model... (this may take a moment)');
        modelStatus.textContent = 'Model status: Loading...';
        
        // Create the sentiment analysis pipeline
        sentimentPipeline = await pipeline(
            'text-classification',
            'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
            { revision: 'main' }
        );
        
        isModelReady = true;
        updateStatus('Sentiment analysis model is ready!');
        modelStatus.textContent = 'Model status: Ready';
        analyzeButton.disabled = !isReviewsLoaded;
        
    } catch (error) {
        showError(`Failed to load sentiment model: ${error.message}`);
        updateStatus('Failed to load sentiment model.', true);
        modelStatus.textContent = 'Model status: Failed to load';
        isModelReady = false;
        analyzeButton.disabled = true;
    }
}

/**
 * Get a random review from the loaded reviews
 */
function getRandomReview() {
    if (reviews.length === 0) {
        throw new Error('No reviews available. Please upload a file or use sample reviews.');
    }
    
    const randomIndex = Math.floor(Math.random() * reviews.length);
    return reviews[randomIndex];
}

/**
 * Analyze the sentiment of a given text using the Transformers.js pipeline
 */
async function analyzeSentiment(text) {
    if (!sentimentPipeline) {
        throw new Error('Sentiment model not loaded.');
    }
    
    // Run the sentiment analysis
    const result = await sentimentPipeline(text);
    
    // The pipeline returns an array of objects, take the first (most confident) result
    if (!result || !Array.isArray(result) || result.length === 0) {
        throw new Error('Invalid response from sentiment analysis model.');
    }
    
    const topResult = result[0];
    
    return {
        label: topResult.label,
        score: topResult.score
    };
}

/**
 * Map the model output to our sentiment categories (positive, negative, neutral)
 */
function categorizeSentiment(label, score) {
    const normalizedLabel = label.toUpperCase();
    
    // Model typically returns "POSITIVE" or "NEGATIVE"
    if (normalizedLabel === 'POSITIVE' && score > 0.5) {
        return 'positive';
    } else if (normalizedLabel === 'NEGATIVE' && score > 0.5) {
        return 'negative';
    } else {
        return 'neutral';
    }
}

/**
 * Update the UI with sentiment results
 */
function updateSentimentUI(sentimentCategory, label, score) {
    // Show the result container
    resultContainer.style.display = 'flex';
    
    // Update sentiment icon and styling
    sentimentIcon.className = 'sentiment-icon fas ';
    
    switch (sentimentCategory) {
        case 'positive':
            sentimentIcon.classList.add('fa-thumbs-up', 'positive');
            sentimentLabel.textContent = 'POSITIVE';
            sentimentLabel.className = 'sentiment-label positive';
            break;
        case 'negative':
            sentimentIcon.classList.add('fa-thumbs-down', 'negative');
            sentimentLabel.textContent = 'NEGATIVE';
            sentimentLabel.className = 'sentiment-label negative';
            break;
        default:
            sentimentIcon.classList.add('fa-question-circle', 'neutral');
            sentimentLabel.textContent = 'NEUTRAL';
            sentimentLabel.className = 'sentiment-label neutral';
    }
    
    // Update confidence score
    const confidencePercent = (score * 100).toFixed(1);
    confidence.textContent = `${confidencePercent}% confidence`;
}

/**
 * Handle the analyze button click
 */
async function handleAnalyzeClick() {
    // Reset UI states
    hideError();
    
    // Validate prerequisites
    if (!isReviewsLoaded) {
        showError('No reviews loaded. Please upload a file or use sample reviews.');
        return;
    }
    
    if (!isModelReady) {
        showError('Sentiment model not ready yet. Please wait.');
        return;
    }
    
    try {
        // Disable button and show loading state
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Analyzing...';
        
        // Get and display random review
        const randomReview = getRandomReview();
        reviewDisplay.textContent = randomReview;
        reviewDisplay.classList.remove('empty');
        
        updateStatus('Analyzing sentiment...');
        
        // Analyze sentiment
        const sentimentResult = await analyzeSentiment(randomReview);
        
        // Categorize sentiment
        const sentimentCategory = categorizeSentiment(
            sentimentResult.label, 
            sentimentResult.score
        );
        
        // Update UI with results
        updateSentimentUI(
            sentimentCategory, 
            sentimentResult.label, 
            sentimentResult.score
        );
        
        updateStatus('Analysis complete!');
        
    } catch (error) {
        showError(`Analysis failed: ${error.message}`);
        updateStatus('Analysis failed.', true);
    } finally {
        // Re-enable button
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = '<i class="fas fa-random"></i> Analyze Random Review';
    }
}

/**
 * Handle file upload
 */
async function handleFileUpload(event) {
    const file = event.target.files[0];
    
    if (!file) {
        return;
    }
    
    try {
        hideError();
        updateStatus(`Loading reviews from ${file.name}...`);
        analyzeButton.disabled = true;
        
        await loadReviewsFromFile(file);
        updateReviewsUI();
        
    } catch (error) {
        showError(`Failed to load file: ${error.message}`);
        updateStatus('File loading failed.', true);
        reviews = [];
        updateReviewsUI();
    }
}

/**
 * Handle sample reviews button click
 */
function handleSampleReviewsClick() {
    try {
        hideError();
        updateStatus('Loading sample reviews...');
        
        const count = loadSampleReviews();
        updateReviewsUI();
        
        updateStatus(`Loaded ${count} sample reviews. Ready to analyze!`);
        
    } catch (error) {
        showError(`Failed to load sample reviews: ${error.message}`);
        updateStatus('Failed to load sample reviews.', true);
    }
}

/**
 * Initialize the application when the DOM is fully loaded
 */
async function initializeApp() {
    try {
        // Set up event listeners
        analyzeButton.addEventListener('click', handleAnalyzeClick);
        tsvFileInput.addEventListener('change', handleFileUpload);
        useSamplesButton.addEventListener('click', handleSampleReviewsClick);
        
        analyzeButton.disabled = true;
        
        // Initialize the model
        updateStatus('Starting application... Loading AI model...');
        await initializeModel();
        
        // Show that we're ready for file upload or samples
        updateStatus('Application ready! Upload a TSV file or use sample reviews to begin.');
        
    } catch (error) {
        showError(`Failed to initialize application: ${error.message}`);
        updateStatus('Initialization failed.', true);
    }
}

// Start the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);