document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://127.0.0.1:8000';

    // --- DOM Elements ---
    const sessionIdInput = document.getElementById('session-id-input');
    const changeUserBtn = document.getElementById('change-user-btn');
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    
    const recommendationsContainer = document.getElementById('recommendations-container');
    const searchResultsContainer = document.getElementById('search-results-container');
    const searchResultsSection = document.getElementById('search-results-section');
    const historyLog = document.getElementById('history-log');

    // --- State ---
    let currentSessionId = '00002ee56cb766213929ea878c38ab1c1e91e688e3a58d7d73c8931db44a7fa0'; // A user with some history
    
    // --- Functions ---

    /**
     * Renders product cards into a specified container.
     * @param {HTMLElement} container - The container to render cards into.
     * @param {Array} products - Array of product objects from the API.
     */
    function renderProductCards(container, products) {
        container.innerHTML = '';
        if (!products || products.length === 0) {
            container.innerHTML = '<p>No products found. Try another search or user!</p>';
            return;
        }

        products.forEach(product => {
            const cardLink = document.createElement('a');
            cardLink.href = product.link.startsWith('#') ? 'javascript:void(0);' : product.link;
            if (!product.link.startsWith('#')) {
                cardLink.target = '_blank'; // Open external links in new tab
            }
            
            cardLink.classList.add('product-card');
            cardLink.dataset.productTitle = product.title;
            
            const priceHTML = product.price ? `<span class="product-price">$${product.price.toFixed(2)}</span>` : '';
            const sourceHTML = product.source ? `<span class="product-source">${product.source}</span>` : '';

            cardLink.innerHTML = `
                <div class="product-image" style="background-image: url('${product.thumbnail}');"></div>
                <div class="product-info">
                    <h3 class="product-name">${product.title}</h3>
                    <div class="product-details">
                        ${priceHTML}
                        ${sourceHTML}
                    </div>
                </div>
            `;
            
            // Track click event
            cardLink.addEventListener('click', async () => {
                await trackEvent('click', product.title);
                // After tracking, refresh recommendations AND history
                await fetchRecommendations();
                await fetchHistory();
            });

            container.appendChild(cardLink);
        });
    }

    /**
     * Fetches recommendations for the current user.
     */
    async function fetchRecommendations() {
        if (!currentSessionId) {
            alert('Please enter a Session ID.');
            return;
        }
        recommendationsContainer.innerHTML = '<p>Loading recommendations...</p>';
        try {
            const response = await fetch(`${API_BASE_URL}/recommend/${currentSessionId}`);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to fetch recommendations');
            }
            const data = await response.json();
            renderProductCards(recommendationsContainer, data.recommendations);
        } catch (error) {
            console.error('Error fetching recommendations:', error);
            recommendationsContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    }

    /**
     * Fetches and renders the user's action history.
     */
    async function fetchHistory() {
        if (!currentSessionId) return;
        try {
            const response = await fetch(`${API_BASE_URL}/history/${currentSessionId}`);
            if (!response.ok) {
                throw new Error('Failed to fetch history');
            }
            const data = await response.json();
            renderHistory(data.history);
        } catch (error) {
            console.error('Error fetching history:', error);
            historyLog.innerHTML = `<li>Error loading history.</li>`;
        }
    }

    /**
     * Renders the history log.
     * @param {Array} history - Array of history event objects.
     */
    function renderHistory(history) {
        historyLog.innerHTML = '';
        if (!history || history.length === 0) {
            historyLog.innerHTML = '<li>No actions recorded yet.</li>';
            return;
        }
        // Show latest actions first
        history.slice().reverse().forEach(event => {
            const logItem = document.createElement('li');
            logItem.innerHTML = `<span class="event-type">${event.event_type}</span> ${event.value}`;
            historyLog.appendChild(logItem);
        });
    }

    /**
     * Performs a product search.
     */
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) {
            alert('Please enter a search term.');
            return;
        }
        
        searchResultsSection.classList.remove('hidden');
        searchResultsContainer.innerHTML = `<p>Searching for "${query}"...</p>`;
        
        try {
            const url = `${API_BASE_URL}/search?q=${encodeURIComponent(query)}&session_id=${encodeURIComponent(currentSessionId)}`;
            const response = await fetch(url);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to perform search');
            }
            const data = await response.json();
            renderProductCards(searchResultsContainer, data.recommendations);
            
            // After a search, refresh recommendations and history to reflect the new interest
            setTimeout(async () => {
                await fetchRecommendations();
                await fetchHistory();
            }, 500);

        } catch (error) {
            console.error('Error performing search:', error);
            searchResultsContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        }
    }

    /**
     * Tracks a user event (e.g., click, search).
     * @param {string} eventType - The type of event ('click', 'search').
     * @param {string} value - The associated value (e.g., product title, search query).
     */
    async function trackEvent(eventType, value) {
        if (!currentSessionId) return;

        try {
            await fetch(`${API_BASE_URL}/events`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    event_type: eventType,
                    value: value
                }),
            });
            console.log(`Tracked event: ${eventType} - ${value}`);
        } catch (error) {
            console.error('Error tracking event:', error);
        }
    }

    // --- Event Listeners ---
    changeUserBtn.addEventListener('click', () => {
        const newSessionId = sessionIdInput.value.trim();
        if (newSessionId) {
            currentSessionId = newSessionId;
            document.querySelector('header h1').textContent = `Store for ${currentSessionId.substring(0, 8)}...`;
            searchResultsSection.classList.add('hidden'); // Hide search results on user change
            fetchRecommendations();
            fetchHistory(); // Also fetch history for the new user
        } else {
            alert('Please enter a valid Session ID.');
        }
    });

    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            performSearch();
        }
    });

    // --- Initial Load ---
    sessionIdInput.value = currentSessionId;
    document.querySelector('header h1').textContent = `Store for ${currentSessionId.substring(0, 8)}...`;
    fetchRecommendations();
    fetchHistory();
}); 