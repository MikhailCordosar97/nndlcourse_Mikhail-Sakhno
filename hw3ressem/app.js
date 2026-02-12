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

// ----- НОВЫЕ элементы для действия -----
const actionResult = document.getElementById('actionResult');
const actionMessage = document.getElementById('actionMessage');
const actionButton = document.getElementById('actionButton');
const actionButtonText = document.getElementById('actionButtonText');

// ----- Конфигурация -----
const GOOGLE_APPS_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbxaVf_U37okJHJfTN0mRJXk1awTWxxYURXZVD0BRSazm0U2vwEcDX2IKwTzq0QmRKGp/exec'; // ваш URL

// ----- Состояние приложения -----
let reviews = [];
let sentimentPipeline = null;
let isModelReady = false;
let isReviewsLoaded = false;

// ----- Резервные отзывы -----
const FALLBACK_REVIEWS = [ /* ... без изменений ... */ ];

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

// ========== ЗАГРУЗКА ОТЗЫВОВ (без изменений) ==========
async function loadReviewsFromTSV() { /* ... полностью как в вашем коде ... */ }
function loadFallbackReviews() { /* ... полностью как в вашем коде ... */ }

// ========== ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ==========
async function initializeModel() { /* ... без изменений ... */ }

// ========== АНАЛИЗ СЕНТИМЕНТА ==========
function getRandomReview() { /* ... без изменений ... */ }
async function analyzeSentiment(text) { /* ... без изменений ... */ }
function categorizeSentiment(label, score) { /* ... без изменений ... */ }
function updateSentimentUI(sentimentCategory, label, score) { /* ... без изменений ... */ }

// ========== НОВОЕ: БИЗНЕС-ЛОГИКА (ИЗ ЗАДАНИЯ) ==========
/**
 * Преобразует сырой результат модели в бизнес-действие.
 * Возвращает объект с actionCode, uiMessage, uiColor, buttonText, icon.
 */
function determineBusinessAction(confidence, label) {
    // 1. Нормализация в индекс 0..1 (0 = плохо, 1 = хорошо)
    let normalizedScore;
    if (label === 'POSITIVE') {
        normalizedScore = confidence;           // 0.9 → 0.9 (отлично)
    } else if (label === 'NEGATIVE') {
        normalizedScore = 1.0 - confidence;    // 0.9 → 0.1 (ужасно)
    } else {
        normalizedScore = 0.5;                 // fallback (нейтрально)
    }

    // 2. Применение порогов из спецификации
    if (normalizedScore <= 0.4) {
        return {
            actionCode: 'OFFER_COUPON',
            uiMessage: '🚨 We are truly sorry. Please accept this 50% discount coupon.',
            uiColor: '#ef4444',        // красный
            buttonText: 'Get Coupon',
            icon: 'fa-ticket'
        };
    } else if (normalizedScore < 0.7) {
        return {
            actionCode: 'REQUEST_FEEDBACK',
            uiMessage: '📝 Thank you! Could you tell us how we can improve?',
            uiColor: '#6b7280',        // серый
            buttonText: 'Give Feedback',
            icon: 'fa-comment-dots'
        };
    } else {
        return {
            actionCode: 'ASK_REFERRAL',
            uiMessage: '⭐ Glad you liked it! Refer a friend and earn rewards.',
            uiColor: '#3b82f6',        // синий
            buttonText: 'Refer Now',
            icon: 'fa-share-alt'
        };
    }
}

/**
 * Отображает панель действия в UI.
 */
function renderAction(decision) {
    // Показываем панель
    actionResult.style.display = 'block';
    actionResult.classList.add('visible');
    actionResult.style.backgroundColor = decision.uiColor + '15'; // очень светлый фон
    actionResult.style.border = `2px solid ${decision.uiColor}`;
    
    // Сообщение и иконка
    actionMessage.innerHTML = `<i class="fas ${decision.icon}" style="color: ${decision.uiColor}"></i> ${decision.uiMessage}`;
    actionButtonText.textContent = decision.buttonText;
    actionButton.style.color = decision.uiColor;
    actionButton.style.borderColor = decision.uiColor;
    actionButton.onclick = () => {
        alert(`[Simulated] Action executed: ${decision.actionCode}\n${decision.uiMessage}`);
        // В реальном проекте здесь был бы вызов CRM-интеграции
    };
}

// ========== РАСШИРЕННОЕ ЛОГИРОВАНИЕ (добавлено action_taken) ==========
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
            // НОВОЕ ПОЛЕ: действие, предпринятое системой
            action_taken: actionCode
        };
        
        console.log('📤 Sending to Google Sheets:', data);
        await fetch(GOOGLE_APPS_SCRIPT_URL, {
            method: 'POST',
            mode: 'no-cors',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        console.log('✅ Data sent to Google Sheets (action_taken included)');
        showSheetsStatus('Data + action saved to Google Sheets');
        return true;
    } catch (error) {
        console.error('❌ Failed to send to Google Sheets:', error);
        showSheetsStatus('Failed to save to Google Sheets', true);
        return false;
    }
}

// ========== ОБРАБОТЧИК КЛИКА (ОСНОВНОЙ) ==========
async function handleAnalyzeClick() {
    hideError();
    if (!isReviewsLoaded) { showError('Reviews not loaded yet.'); return; }
    if (!isModelReady) { showError('Sentiment model not ready yet.'); return; }
    
    try {
        analyzeButton.disabled = true;
        analyzeButton.innerHTML = '<span class="loading"></span> Analyzing...';
        
        const randomReview = getRandomReview();
        reviewDisplay.textContent = randomReview;
        reviewDisplay.classList.remove('empty');
        updateStatus('Analyzing sentiment...');
        
        // 1. Анализ тональности
        const sentimentResult = await analyzeSentiment(randomReview);
        const sentimentCategory = categorizeSentiment(sentimentResult.label, sentimentResult.score);
        updateSentimentUI(sentimentCategory, sentimentResult.label, sentimentResult.score);
        
        // 2. БИЗНЕС-ЛОГИКА: принимаем решение
        const decision = determineBusinessAction(sentimentResult.score, sentimentResult.label);
        console.log('🧠 Decision:', decision);
        
        // 3. Отображаем действие в UI
        renderAction(decision);
        
        // 4. Отправляем ВСЕ данные (включая action_taken) в Google Sheets
        await sendToGoogleSheets(randomReview, sentimentResult, sentimentCategory, decision.actionCode);
        
        updateStatus('Analysis complete → Action taken → Logged');
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
    await loadReviewsFromTSV();
    await initializeModel();
    updateStatus('Application ready! Click "Analyze Random Review" to start.');
}
document.addEventListener('DOMContentLoaded', initializeApp);