console.log('Checking if Plotly is loaded...');
if (typeof Plotly === 'undefined') {
    console.error('Plotly is not loaded!');
} else {
    console.log('Plotly is loaded successfully:', Plotly.version);
}

let lastPlotData = null; // Store the last data used for plotting
let agentMode = false;

const modeToggleBtn = document.getElementById('mode-toggle');
modeToggleBtn.onclick = function() {
    agentMode = !agentMode;
    if (agentMode) {
        modeToggleBtn.textContent = 'Agent';
        modeToggleBtn.classList.add('agent');
    } else {
        modeToggleBtn.textContent = 'Ask';
        modeToggleBtn.classList.remove('agent');
    }
};

const themeBtn = document.getElementById('toggle-theme');
themeBtn.onclick = function() {
    document.body.classList.toggle('dark-mode');
    themeBtn.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
    if (lastPlotData) {
        plotGraph(lastPlotData);
    }
};

const form = document.getElementById('queryForm');
form.onsubmit = async function(e) {
    e.preventDefault();
    const query = document.getElementById('userQuery').value;
    const mode = document.getElementById('mode-toggle').textContent === 'Ask Mode' ? 'ask' : 'agent';
    const endpoint = mode === 'ask' ? '/analyze' : '/analyze_agent';
    
    console.log('Submitting query:', { query, mode, endpoint });
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        console.log('Received response:', data);
        
        if (data.status === 'success') {
            if (data.plots) {
                console.log('Plotting data:', data.plots);
                plotGraph(data.plots);
            } else {
                console.warn('No plot data received');
                document.getElementById('plot').innerHTML = '<div style="padding: 20px;">No plot data available</div>';
            }
            
            if (data.insights) {
                document.getElementById('insights').innerHTML = data.insights;
            }
        } else {
            console.error('Error response:', data);
            document.getElementById('plot').innerHTML = `<div style="color: red; padding: 20px;">${data.message || 'Error occurred'}</div>`;
        }
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('plot').innerHTML = `<div style="color: red; padding: 20px;">Error: ${error.message}</div>`;
    }
};

function renderInsights(insightsText) {
    const lines = insightsText.split('\n');
    let html = '';
    let inList = false;

    lines.forEach(line => {
        // Remove all asterisks except those used for bolding
        let cleanedLine = line.replace(/^\s*\*\s*/, ''); // Remove leading bullet asterisk
        // Remove lines that are just a bullet (empty after cleaning)
        if (/^\s*\*\s*$/.test(line)) return;

        // If the line is empty after cleaning, skip it
        if (cleanedLine.trim() === '') {
            if (inList) {
                html += '</ul>';
                inList = false;
            }
            return;
        }

        // If the line starts with a bullet, add to list
        if (/^\s*\*/.test(line)) {
            if (!inList) {
                html += '<ul class="insights-bullet-list">';
                inList = true;
            }
            // Bold and color text between double asterisks
            let bulletText = cleanedLine.replace(/\*\*(.+?)\*\*/g, '<span class="insights-bold">$1</span>');
            // Remove any stray asterisks
            bulletText = bulletText.replace(/\*/g, '');
            html += `<li class="insights-bullet">${bulletText}</li>`;
            return;
        }

        // For non-bullet lines, close list if open
        if (inList) {
            html += '</ul>';
            inList = false;
        }
        // Bold and color text between double asterisks
        let text = cleanedLine.replace(/\*\*(.+?)\*\*/g, '<span class="insights-bold">$1</span>');
        // Remove any stray asterisks
        text = text.replace(/\*/g, '');
        html += `<div class="insights-text">${text}</div>`;
    });

    if (inList) html += '</ul>';
    document.getElementById('insights').innerHTML = html;
}

function getPlotlyLayout(isDark) {
    return {
        title: 'Trend & Forecast',
        xaxis: {title: 'Date', color: isDark ? '#e0e0e0' : '#222', gridcolor: isDark ? '#444' : '#e0e0e0'},
        yaxis: {title: 'Value', color: isDark ? '#e0e0e0' : '#222', gridcolor: isDark ? '#444' : '#e0e0e0'},
        plot_bgcolor: isDark ? '#23272b' : '#f8f8ff',
        paper_bgcolor: isDark ? '#23272b' : '#fff',
        font: {color: isDark ? '#e0e0e0' : '#222'},
        legend: {orientation: "h", y: -0.2}
    };
}

function plotGraph(data) {
    console.log('Plotting graph with data:', data);
    
    // Verify Plotly is available
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded! Cannot create plots.');
        document.getElementById('plot').innerHTML = '<div style="color: red; padding: 20px;">Error: Plotly library not loaded</div>';
        return;
    }
    
    // Clear previous plots
    const plotContainer = document.getElementById('plot');
    plotContainer.innerHTML = '';
    
    // Create grid container
    const gridContainer = document.createElement('div');
    gridContainer.className = 'plot-grid';
    plotContainer.appendChild(gridContainer);
    
    // Base layout for all plots
    const baseLayout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: {
            color: document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#222'
        },
        margin: { t: 40, r: 20, b: 40, l: 60 },
        showlegend: true,
        legend: {
            x: 0.5,
            y: 1.1,
            xanchor: 'center',
            orientation: 'h'
        },
        xaxis: {
            showgrid: true,
            gridcolor: document.body.classList.contains('dark-mode') ? '#444' : '#ddd',
            zeroline: false
        },
        yaxis: {
            showgrid: true,
            gridcolor: document.body.classList.contains('dark-mode') ? '#444' : '#ddd',
            zeroline: false
        }
    };
    
    // Plot each graph
    Object.entries(data).forEach(([key, plotData]) => {
        console.log(`Creating plot for ${key}:`, plotData);
        
        // Create card for each plot
        const card = document.createElement('div');
        card.className = 'plot-card';
        gridContainer.appendChild(card);
        
        const plotDiv = document.createElement('div');
        plotDiv.style.width = '100%';
        plotDiv.style.height = '380px';
        card.appendChild(plotDiv);
        
        const layout = {
            ...baseLayout,
            title: {
                text: key.charAt(0).toUpperCase() + key.slice(1) + ' Analysis',
                font: {
                    size: 16,
                    color: document.body.classList.contains('dark-mode') ? '#e0e0e0' : '#222'
                }
            }
        };
        
        try {
            console.log(`Attempting to create plot for ${key} with data:`, plotData.data);
            console.log(`Using layout:`, layout);
            
            Plotly.newPlot(plotDiv, plotData.data, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                modeBarButtonsToRemove: ['lasso2d', 'select2d']
            }).then(() => {
                console.log(`Successfully created plot for ${key}`);
            }).catch(error => {
                console.error(`Error in plot creation for ${key}:`, error);
                plotDiv.innerHTML = `<div style=\"color: red; padding: 20px;\">Error creating plot: ${error.message}</div>`;
            });
        } catch (error) {
            console.error(`Error creating plot for ${key}:`, error);
            plotDiv.innerHTML = `<div style=\"color: red; padding: 20px;\">Error creating plot: ${error.message}</div>`;
        }
    });
}
