import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js";

// ----- DOM элементы -----
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
const googleSheetsStatus = document.getElementById('googleSheetsStatus');
const sheetsStatusText = document.getElementById('sheetsStatusText');

// ----- элементы бизнес-действия -----
const actionResult = document.getElementById('actionResult');
const actionMessage = document.getElementById('actionMessage');
const actionButton = document.getElementById('actionButton');
const actionButtonText = document.getElementById('actionButtonText');

// ----- Конфигурация -----
const GOOGLE_APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbyxGH4hfMNfrp8UhmXtK20-Q7eoEbxVDMX4DE7wHjlmMS2OnTiwoK_i8q5HVMjzqvW4/exec';

// ----- Состояние приложения -----
let reviews = [];
let sentimentPipeline = null;
let isModelReady = false;
let isReviewsLoaded = false;

// ----- Резервные отзывы -----
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

// ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
function updateStatus(message, isError = false) {
    statusText.innerHTML = message;
    console[isError ? 'error' : 'log'](message);
}
function showError(message) {
    errorText.textContent = message;
    errorPanel.style.display = 'block';
    console.error(message);
}
function hideError() { errorPanel.style.display = 'none'; }
function showSheetsStatus(message, isError = false) {
    sheetsStatusText.textContent = message;
    googleSheetsStatus.style.display = 'block';
    googleSheetsStatus.classList.toggle('error', isError);
    setTimeout(() => { googleSheetsStatus.style.display = 'none'; }, 5000);
}

// ========== ЗАГРУЗКА ОТЗЫВОВ (быстро) ==========
async function loadReviewsFromTSV() {
    try {
        updateStatus('Loading reviews from reviews_test.tsv...');
        reviewDisplay.textContent = '📂 Loading review file...';
        const response = await fetch('reviews_test.tsv');
        if (!response.ok) {
            if (response.status === 404) throw new Error('File not found');
            throw new Error(`HTTP ${response.status}`);
        }
        const tsvContent = await response.text();
        const result = Papa.parse(tsvContent, { header: true, delimiter: '\t', skipEmptyLines: true });
        reviews = result.data
            .map(row => {
                const textKey = Object.keys(row).find(key => 
                    key.toLowerCase() === 'text' || key.toLowerCase() === 'review'
                );
                return textKey ? row[textKey] : Object.values(row)[0];
            })
            .filter(text => text && typeof text === 'string' && text.trim().length > 0)
            .map(text => text.trim());
        if (reviews.length === 0) throw new Error('No valid reviews');
        isReviewsLoaded = true;
        reviewStats.textContent = `Reviews loaded: ${reviews.length}`;
        updateStatus(`✅ Loaded ${reviews.length} reviews from file.`);
        return true;
    } catch (error) {
        console.warn('TSV load failed, using samples:', error.message);
        return loadFallbackReviews();
    }
}

function loadFallbackReviews() {
    reviews = [...FALLBACK_REVIEWS];
    isReviewsLoaded = true;
    reviewStats.textContent = `Reviews loaded: ${reviews.length} (sample)`;
    updateStatus(`📋 Using ${reviews.length} sample reviews.`);
    return true;
}

// ========== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ (с таймером и подсказками) ==========
async function initializeModel() {
    try {
        // Сразу меняем текст в области отзыва, чтобы пользователь знал, что модель грузится
        reviewDisplay.textContent = '🤖 Loading AI model... This may take up to 2 minutes on first run. Please wait.';
        updateStatus('Loading sentiment analysis model (first time can be slow)...');
        modelStatus.textContent = 'Model status: ⏳ Loading...';

        // --- Таймер для подсказки, если загрузка затянулась ---
        let slowTimer = setTimeout(() => {
            reviewDisplay.textContent = '⏱️ Still loading the AI model... If this takes more than 3 minutes, please check your internet or refresh the page.';
            updateStatus('Model download is taking longer than expected – please be patient.', true);
        }, 30000); // 30 секунд

        // Загружаем модель
        sentimentPipeline = await pipeline(
            'text-classification',
            'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
            { revision: 'main' }
        );

        clearTimeout(slowTimer); // отменяем таймер, если модель загрузилась

        isModelReady = true;
        updateStatus('✅ Sentiment analysis model is ready!');
        modelStatus.textContent = 'Model status: ✅ Ready';
        reviewDisplay.textContent = isReviewsLoaded 
            ? 'Ready! Click "Analyze Random Review" to start.'
            : 'Reviews not loaded yet, but model is ready.';

        if (isReviewsLoaded) analyzeButton.disabled = false;

    } catch (error) {
        console.error('Model initialization failed:', error);
        showError(`Failed to load sentiment model: ${error.message}`);
        updateStatus('❌ Model failed to load.', true);
        modelStatus.textContent = 'Model status: ❌ Failed';
        reviewDisplay.textContent = '⚠️ AI model could not be loaded. You can still analyze using sample reviews (click button below).';
        isModelReady = false;
        analyzeButton.disabled = true; // блокируем, так как без модели анализ невозможен
    }
}

// ========== АНАЛИЗ СЕНТИМЕНТА ==========
function getRandomReview() {
    if (reviews.length === 0) throw new Error('No reviews available.');
    return reviews[Math.floor(Math.random() * reviews.length)];
}
async function analyzeSentiment(text) {
    if (!sentimentPipeline) throw new Error('Sentiment model not loaded.');
    const result = await sentimentPipeline(text);
    if (!Array.isArray(result) || result.length === 0) throw new Error('Invalid model response');
    return result[0];
}
function categorizeSentiment(label, score) {
    const normalizedLabel = label.toUpperCase();
    if (normalizedLabel === 'POSITIVE' && score > 0.5) return 'positive';
    if (normalizedLabel === 'NEGATIVE' && score > 0.5) return 'negative';
    return 'neutral';
}
function updateSentimentUI(sentimentCategory, label, score) {
    resultContainer.style.display = 'flex';
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
    confidence.textContent = `${(score * 100).toFixed(1)}% confidence`;
}

// ========== БИЗНЕС-ЛОГИКА (ИЗ ЗАДАНИЯ) ==========
function determineBusinessAction(confidence, label) {
    let normalizedScore;
    if (label === 'POSITIVE') normalizedScore = confidence;
    else if (label === 'NEGATIVE') normalizedScore = 1.0 - confidence;
    else normalizedScore = 0.5;

    if (normalizedScore <= 0.4) {
        return {
            actionCode: 'OFFER_COUPON',
            uiMessage: '🚨 We are truly sorry. Please accept this 50% discount coupon.',
            uiColor: '#ef4444',
            buttonText: 'Get Coupon',
            icon: 'fa-ticket'
        };
    } else if (normalizedScore < 0.7) {
        return {
            actionCode: 'REQUEST_FEEDBACK',
            uiMessage: '📝 Thank you! Could you tell us how we can improve?',
            uiColor: '#6b7280',
            buttonText: 'Give Feedback',
            icon: 'fa-comment-dots'
        };
    } else {
        return {
            actionCode: 'ASK_REFERRAL',
            uiMessage: '⭐ Glad you liked it! Refer a friend and earn rewards.',
            uiColor: '#3b82f6',
            buttonText: 'Refer Now',
            icon: 'fa-share-alt'
        };
    }
}

function renderAction(decision) {
    actionResult.style.display = 'block';
    actionResult.classList.add('visible');
    actionResult.style.backgroundColor = decision.uiColor + '15';
    actionResult.style.border = `2px solid ${decision.uiColor}`;
    actionMessage.innerHTML = `<i class="fas ${decision.icon}" style="color: ${decision.uiColor}"></i> ${decision.uiMessage}`;
    actionButtonText.textContent = decision.buttonText;
    actionButton.style.color = decision.uiColor;
    actionButton.style.borderColor = decision.uiColor;
    actionButton.onclick = () => alert(`[Simulated] Action: ${decision.actionCode}\n${decision.uiMessage}`);
}

// ========== РАСШИРЕННОЕ ЛОГИРОВАНИЕ (action_taken) ==========
async function sendToGoogleSheets(reviewText, sentimentResult, sentimentCategory, actionCode) {
    try {
        const data = {
            timestamp: new Date().toISOString(),
            review: reviewText,
            sentiment: {
                label: sentimentResult.label,
                score: sentimentResult.score,
                category: sentimentCategory
            },
            meta: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language,
                screenResolution: `${window.screen.width}x${window.screen.height}`,
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
                reviewsCount: reviews.length,
                model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
                timestampClient: new Date().getTime()
            },
            action_taken: actionCode
        };
        console.log('📤 Sending to Google Sheets:', data);
        await fetch(GOOGLE_APPS_SCRIPT_URL, {
            method: 'POST',
            mode: 'no-cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        console.log('✅ Data + action logged');
        showSheetsStatus('Data + action saved to Google Sheets');
        return true;
    } catch (error) {
        console.error('❌ Google Sheets error:', error);
        showSheetsStatus('Failed to save to Google Sheets', true);
        return false;
    }
}

// ========== ОБРАБОТЧИК КЛИКА ==========
async function handleAnalyzeClick() {
    hideError();
    if (!isReviewsLoaded) { showError('Reviews not loaded yet.'); return; }
    if (!isModelReady) { 
        showError('Sentiment model is not ready. Please wait or refresh if it takes too long.'); 
        return; 
    }
    try {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Analyzing...';
        
        const randomReview = getRandomReview();
        reviewDisplay.textContent = randomReview;
        reviewDisplay.classList.remove('empty');
        updateStatus('Analyzing sentiment...');
        
        const sentimentResult = await analyzeSentiment(randomReview);
        const sentimentCategory = categorizeSentiment(sentimentResult.label, sentimentResult.score);
        updateSentimentUI(sentimentCategory, sentimentResult.label, sentimentResult.score);
        
        const decision = determineBusinessAction(sentimentResult.score, sentimentResult.label);
        console.log('🧠 Decision:', decision);
        renderAction(decision);
        
        await sendToGoogleSheets(randomReview, sentimentResult, sentimentCategory, decision.actionCode);
        updateStatus('✅ Analysis complete → Action taken → Logged');
    } catch (error) {
        showError(`Analysis failed: ${error.message}`);
        updateStatus('Analysis failed.', true);
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = '<i class="fas fa-random"></i> Analyze Random Review';
    }
}

// ========== ИНИЦИАЛИЗАЦИЯ ==========
async function initializeApp() {
    analyzeButton.addEventListener('click', handleAnalyzeClick);
    analyzeButton.disabled = true;
    
    updateStatus('Starting application...');
    reviewDisplay.textContent = '📂 Loading review file...';
    
    // Сначала быстро загружаем отзывы
    await loadReviewsFromTSV();
    
    // Затем запускаем загрузку модели (с таймером и подсказками)
    await initializeModel();
    
    // Финальное сообщение
    if (isModelReady && isReviewsLoaded) {
        updateStatus('✅ Ready! Click "Analyze Random Review" to start.');
    } else if (!isModelReady) {
        updateStatus('⚠️ Model failed to load – analysis unavailable.', true);
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);
