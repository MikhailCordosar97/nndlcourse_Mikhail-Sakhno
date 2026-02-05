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
const fallbackPanel = document.getElementById('fallbackPanel');
const useFallbackButton = document.getElementById('useFallbackButton');

// Application state
let reviews = [];
let sentimentPipeline = null;
let isModelReady = false;
let isReviewsLoaded = false;

// Fallback sample reviews if TSV file cannot be loaded
const FALLBACK_REVIEWS = [
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
 * Load and parse the TSV file containing reviews from the server
 */
async function loadReviewsFromTSV() {
    try {
        updateStatus('Loading reviews from reviews_test.tsv...');
        
        const response = await fetch('reviews_test.tsv');
        
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('File reviews_test.tsv not found. Please ensure it exists in the same directory.');
            }
            throw new Error(`Failed to load TSV file: ${response.status} ${response.statusText}`);
        }
        
        const tsvContent = await response.text();
        
        // Parse TSV using Papa Parse
        const result = Papa.parse(tsvContent, {
            header: true,
            delimiter: '\t',
            skipEmptyLines: true,
            dynamicTyping: false
        });
        
        if (result.errors && result.errors.length > 0) {
            console.warn('TSV parsing warnings:', result.errors);
        }
        
        // Extract review texts from the 'text' column
        reviews = result.data
            .map(row => {
                // Try to find text column (case-insensitive)
                const textKey = Object.keys(row).find(key => 
                    key.toLowerCase() === 'text' || key.toLowerCase() === 'review'
                );
                return textKey ? row[textKey] : row.text;
            })
            .filter(text => text && typeof text === 'string' && text.trim().length > 0)
            .map(text => text.trim());
        
        if (reviews.length === 0) {
            // If no 'text' column found, try using first column
            reviews = result.data
                .map(row => {
                    const firstColumn = Object.values(row)[0];
                    return typeof firstColumn === 'string' ? firstColumn : null;
                })
                .filter(text => text && text.trim().length > 0)
                .map(text => text.trim());
        }
        
        if (reviews.length === 0) {
            throw new Error('No valid reviews found in the TSV file. Please check the format.');
        }
        
        isReviewsLoaded = true;
        reviewStats.textContent = `Reviews loaded: ${reviews.length}`;
        updateStatus(`Successfully loaded ${reviews.length} reviews from reviews_test.tsv.`);
        return true;
        
    } catch (error) {
        console.error('Failed to load TSV:', error);
        return false;
    }
}

/**
 * Load fallback sample reviews
 */
function loadFallbackReviews() {
    reviews = [...FALLBACK_REVIEWS];
    isReviewsLoaded = reviews.length > 0;
    reviewStats.textContent = `Reviews loaded: ${reviews.length} (sample)`;
    updateStatus(`Loaded ${reviews.length} sample reviews.`);
    return reviews.length;
}

/**
 * Update UI after reviews are loaded
 */
function updateUIAfterReviewsLoaded() {
    if (isReviewsLoaded) {
        reviewDisplay.textContent = 'Reviews loaded! Click "Analyze Random Review" to start.';
        reviewDisplay.classList.remove('empty');
        analyzeButton.disabled = !isModelReady;
        fallbackPanel.style.display = 'none';
    } else {
        reviewDisplay.textContent = 'Failed to load reviews. Please check console.';
        reviewDisplay.classList.add('empty');
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
        
        // Enable analyze button if reviews are loaded
        if (isReviewsLoaded) {
            analyzeButton.disabled = false;
        }
        
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
        throw new Error('No reviews available.');
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
        showError('Reviews not loaded yet. Please wait.');
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
 * Handle fallback button click to use sample reviews
 */
function handleFallbackButtonClick() {
    try {
        hideError();
        fallbackPanel.style.display = 'none';
        
        loadFallbackReviews();
        updateUIAfterReviewsLoaded();
        
        if (isModelReady) {
            analyzeButton.disabled = false;
        }
        
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
        useFallbackButton.addEventListener('click', handleFallbackButtonClick);
        
        analyzeButton.disabled = true;
        
        // Start loading reviews and model in parallel
        updateStatus('Starting application initialization...');
        
        // Try to load reviews from TSV file first
        const tsvLoaded = await loadReviewsFromTSV();
        
        if (!tsvLoaded) {
            // Show fallback option if TSV file not found
            fallbackPanel.style.display = 'block';
            updateStatus('Could not load reviews_test.tsv. You can use sample reviews instead.', true);
        } else {
            updateUIAfterReviewsLoaded();
        }
        
        // Initialize the model
        await initializeModel();
        
        // Final status update
        if (tsvLoaded && isModelReady) {
            updateStatus('Application ready! Click "Analyze Random Review" to start.');
        } else if (!tsvLoaded && isModelReady) {
            updateStatus('Model ready. Please use sample reviews or check reviews_test.tsv file.');
        }
        
    } catch (error) {
        showError(`Failed to initialize application: ${error.message}`);
        updateStatus('Initialization failed.', true);
    }
}

// Start the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);