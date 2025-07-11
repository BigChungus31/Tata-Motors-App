// NEW app.js

// Global variables
let selectedMaterial = null;
let searchTimeout = null;
let searchCache = new Map();
let dataLoaded = false; // Track if Supabase data is loaded

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    setupSearch();
    setupPrediction();
    initializeTooltips();
});

// Material search functionality - Fixed
function setupSearch() {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) return;
    
    // Initially disable search until data is loaded
    searchInput.disabled = false;
    searchInput.placeholder = 'Search materials...';
    
    searchInput.addEventListener('input', (e) => {
        
        clearTimeout(searchTimeout);
        const query = e.target.value.trim();
        
        if (!query) {
            clearSearchResults();
            return;
        }
        
        // Debounced search
        searchTimeout = setTimeout(() => {
            searchMaterials(query);
        }, 250);
    });
    
    // Clear results when input is cleared
    searchInput.addEventListener('keyup', (e) => {
        if (e.key === 'Escape') {
            clearSearchResults();
            e.target.value = '';
        }
    });
}

function searchMaterials(query) {
    
    // Check cache first
    if (searchCache.has(query)) {
        displaySearchResults(searchCache.get(query));
        return;
    }
    
    showSpinner(true);
    
    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Search failed: HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const materials = data.materials || [];
        // Cache the results
        searchCache.set(query, materials);
        displaySearchResults(materials);
    })
    .catch(error => {
        console.error('Search failed:', error);
        showToast(error.message || 'Search request failed. Please try again.', 'error');
        clearSearchResults();
    })
    .finally(() => {
        showSpinner(false);
    });
}

function displaySearchResults(materials) {
    const tbody = document.getElementById('searchResults')?.querySelector('tbody');
    if (!tbody) return;
    
    tbody.innerHTML = '';
    
    if (materials.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" class="text-center text-muted">No materials found</td></tr>';
        return;
    }
    
    // Fragment for better performance
    const fragment = document.createDocumentFragment();
    
    materials.forEach(material => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td><code>${escapeHtml(material.material_number)}</code></td>
            <td title="${escapeHtml(material.description)}">${escapeHtml(material.description.substring(0, 40))}${material.description.length > 40 ? '...' : ''}</td>
            <td><button class="btn btn-sm btn-primary" onclick="selectMaterial('${escapeHtml(material.material_number)}', '${escapeHtml(material.description)}')">Select</button></td>
        `;
        fragment.appendChild(row);
    });
    
    tbody.appendChild(fragment);
}

function clearSearchResults() {
    const tbody = document.getElementById('searchResults')?.querySelector('tbody');
    if (tbody) {
        tbody.innerHTML = '';
    }
}

function selectMaterial(materialNumber, description) {
    selectedMaterial = materialNumber;
    const selectedInput = document.getElementById('selectedMaterial');
    if (selectedInput) {
        selectedInput.value = `${materialNumber} - ${description}`;
    }
    
    // Auto-focus quantity input for better UX
    const qtyInput = document.getElementById('currentQty');
    if (qtyInput) {
        qtyInput.focus();
        qtyInput.select();
    }
    
    // Clear search results after selection
    clearSearchResults();
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.value = '';
    }
}

// Prediction functionality - Enhanced
function setupPrediction() {
    const predictionForm = document.getElementById('predictionForm');
    if (!predictionForm) return;
    
    predictionForm.addEventListener('submit', (e) => {
        e.preventDefault();
        predictInventory();
    });
    
    // Add Enter key support for quantity input
    const qtyInput = document.getElementById('currentQty');
    if (qtyInput) {
        qtyInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                predictInventory();
            }
        });
        
        // Input validation
        qtyInput.addEventListener('input', (e) => {
            const value = e.target.value;
            if (value && (isNaN(value) || parseFloat(value) < 0)) {
                e.target.setCustomValidity('Please enter a valid positive number');
            } else {
                e.target.setCustomValidity('');
            }
        });
    }
}

function predictInventory() {
    const currentQty = document.getElementById('currentQty')?.value;
    
    if (!selectedMaterial) {
        showToast('Please select a material first', 'error');
        return;
    }
    
    if (!currentQty || isNaN(currentQty) || parseFloat(currentQty) < 0) {
        showToast('Please enter a valid current quantity', 'error');
        return;
    }
    
    showSpinner(true);
    
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            material_number: selectedMaterial,
            current_qty: parseFloat(currentQty)
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Prediction failed: HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        displayPredictionResults(data);
    })
    .catch(error => {
        console.error('Prediction error:', error);
        showToast(error.message || 'Prediction request failed. Please try again.', 'error');
        clearPredictionResults();
    })
    .finally(() => {
        showSpinner(false);
    });
}

function displayPredictionResults(data) {
    const resultsDiv = document.getElementById('predictionResults');
    if (!resultsDiv) return;
    
    if (!data.success) {
        resultsDiv.innerHTML = `<div class="alert alert-danger">${escapeHtml(data.error)}</div>`;
        return;
    }
    
    const prediction = data.prediction;
    const statusClass = getStatusClass(prediction.status);
    
    resultsDiv.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h6 class="mb-0">Inventory Analysis Results</h6>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Current Status</h6>
                        <span class="badge bg-${statusClass} fs-6">${escapeHtml(prediction.status)}</span>
                        <p class="mt-2">Coverage: <strong>${prediction.coverage_months} months</strong></p>
                        ${prediction.shortfall ? `<p class="text-danger">Shortfall: <strong>${prediction.shortfall}</strong></p>` : ''}
                        ${prediction.surplus ? `<p class="text-warning">Surplus: <strong>${prediction.surplus}</strong></p>` : ''}
                    </div>
                    <div class="col-md-6">
                        <h6>Consumption Pattern</h6>
                        <p>Average Monthly: <strong>${prediction.avg_monthly_consumption}</strong></p>
                        <p>Peak (${escapeHtml(prediction.peak_month)}): <strong>${prediction.peak_consumption}</strong></p>
                        <p>Low (${escapeHtml(prediction.low_month)}): <strong>${prediction.low_consumption}</strong></p>
                    </div>
                </div>
                
                ${prediction.busy_months && Object.keys(prediction.busy_months).length > 0 ? `
                <div class="row mt-3">
                    <div class="col-md-6">
                        <h6>High Demand Months</h6>
                        <ul class="list-unstyled">
                            ${Object.entries(prediction.busy_months).map(([month, qty]) => 
                                `<li>${escapeHtml(month)}: <strong>${qty}</strong></li>`
                            ).join('')}
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Low Demand Months</h6>
                        <ul class="list-unstyled">
                            ${Object.entries(prediction.slow_months || {}).map(([month, qty]) => 
                                `<li>${escapeHtml(month)}: <strong>${qty}</strong></li>`
                            ).join('')}
                        </ul>
                    </div>
                </div>
                ` : ''}
                
                <div class="mt-3">
                    <h6>Recommendations</h6>
                    <div class="alert alert-info">
                        ${prediction.recommendation ? escapeHtml(prediction.recommendation) : 'No specific recommendations available'}
                    </div>
                    ${prediction.recommended_order_qty ? `
                        <p><strong>Recommended Order Quantity:</strong> ${prediction.recommended_order_qty}</p>
                    ` : ''}
                </div>
                
                <div class="mt-3">
                    <small class="text-muted">
                        Demand Variability: ${escapeHtml(prediction.cv_interpretation || 'Not available')} 
                        (CV: ${prediction.coefficient_of_variation || 'N/A'})
                    </small>
                </div>
            </div>
        </div>
    `;
}

function clearPredictionResults() {
    const resultsDiv = document.getElementById('predictionResults');
    if (resultsDiv) {
        resultsDiv.innerHTML = '';
    }
}

function getStatusClass(status) {
    const statusMap = {
        'Critical': 'danger',
        'Low': 'warning', 
        'Adequate': 'success',
        'Excess': 'info'
    };
    return statusMap[status] || 'secondary';
}

// Add refresh data function
function refreshData() {
    dataLoaded = false;
    searchCache.clear();
    clearSearchResults();
    clearPredictionResults();
    
}


    showSpinner(true);
    
    fetch('/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
            filename: `material_analysis_${reportType}_${new Date().toISOString().split('T')[0]}.xlsx`,
            report_type: reportType
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Export failed: HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Create download link
            const link = document.createElement('a');
            link.href = `/download/${data.filename}`;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showToast('Export completed successfully!', 'success');
        } else {
            throw new Error(data.error || 'Export failed');
        }
    })
    .catch(error => {
        console.error('Export error:', error);
        showToast(error.message, 'error');
    })
    .finally(() => {
        showSpinner(false);
    });


function exportReport() {
    exportData('full');
}

function exportSummary() {
    exportData('summary');
}

// Utility functions
function showSpinner(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.classList.toggle('d-none', !show);
    }
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('alertToast');
    const toastMessage = document.getElementById('toastMessage');
    
    if (!toast || !toastMessage) {
        console.log(`${type.toUpperCase()}: ${message}`);
        return;
    }
    
    toastMessage.textContent = message;
    
    const typeClasses = {
        'success': 'bg-success text-white',
        'error': 'bg-danger text-white', 
        'warning': 'bg-warning text-dark',
        'info': 'bg-info text-white'
    };
    
    toast.className = `toast ${typeClasses[type] || typeClasses.info}`;
    
    if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: type === 'error' ? 5000 : 3000
        });
        bsToast.show();
    } else {
        toast.style.display = 'block';
        setTimeout(() => {
            toast.style.display = 'none';
        }, type === 'error' ? 5000 : 3000);
    }
}

function escapeHtml(text) {
    if (typeof text !== 'string') return text;
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function initializeTooltips() {
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

function clearAllData() {
    selectedMaterial = null;
    searchCache.clear();
    
    const inputs = ['searchInput', 'selectedMaterial', 'currentQty'];
    inputs.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.value = '';
    });
    
    clearSearchResults();
    clearPredictionResults();
}

// Cart functionality
function addToCart(materialNumber, quantity, unitPrice, description) {
    fetch('/add_to_cart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            material_number: materialNumber,
            quantity: quantity,
            unit_price: unitPrice,
            description: description
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Material added to cart successfully!', 'success');
        } else {
            showToast(data.error || 'Failed to add to cart', 'error');
        }
    })
    .catch(error => {
        console.error('Cart error:', error);
        showToast('Failed to add to cart', 'error');
    });
}

function loadCart() {
    fetch('/get_cart')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayCart(data.cart);
        } else {
            console.error('Failed to load cart:', data.error);
        }
    })
    .catch(error => {
        console.error('Cart load error:', error);
    });
}

function displayCart(cart) {
    const cartContainer = document.getElementById('cartContainer');
    if (!cartContainer) return;
    
    if (Object.keys(cart).length === 0) {
        cartContainer.innerHTML = `
            <div class="text-center py-5">
                <i class="bi bi-cart3 text-muted" style="font-size: 4rem;"></i>
                <h5 class="mt-3 text-muted">Your cart is empty</h5>
                <p class="text-muted">Add materials from your analysis to start building your cart.</p>
                <a href="/dashboard" class="btn btn-primary mt-3">
                    <i class="bi bi-plus-circle me-2"></i>Start Shopping
                </a>
            </div>
        `;
        return;
    }
    
    let cartHtml = '<div class="table-responsive"><table class="table table-striped"><thead><tr><th>Material</th><th>Description</th><th>Quantity</th><th>Unit Price</th><th>Total</th></tr></thead><tbody>';
    let grandTotal = 0;
    
    Object.entries(cart).forEach(([materialNumber, item]) => {
        cartHtml += `
            <tr>
                <td><code>${escapeHtml(materialNumber)}</code></td>
                <td>${escapeHtml(item.description)}</td>
                <td>${item.quantity}</td>
                <td>₹${item.unit_price.toFixed(2)}</td>
                <td>₹${item.total.toFixed(2)}</td>
            </tr>
        `;
        grandTotal += item.total;
    });
    
    cartHtml += `</tbody></table></div>
        <div class="text-end mt-3">
            <h5>Grand Total: ₹${grandTotal.toFixed(2)}</h5>
        </div>`;
    
    cartContainer.innerHTML = cartHtml;
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.getElementById('searchInput');
        if (searchInput && !searchInput.disabled) {
            searchInput.focus();
            searchInput.select();
        }
    }
    
    if (e.key === 'Escape') {
        clearTimeout(searchTimeout);
        showSpinner(false);
    }
    
    // F5 or Ctrl+R to refresh data
    if (e.key === 'F5' || ((e.ctrlKey || e.metaKey) && e.key === 'r')) {
        e.preventDefault();
        refreshData();
    }
});