async function fetchWeightsAndDraw() {
    const numLayers = parseInt(document.getElementById('num-layers').value);
    const layerConfig = document.getElementById('layer-config').value.split(',').map(num => parseInt(num.trim()));

    // Send the layers configuration to the backend to get the weights
    fetch('/get-weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ layers_config: layerConfig })
    })
    .then(response => response.json())  // Convert response to JSON
    .then(data => {
        // Use the 'data.weights' to draw the network
        console.log(data.weights);
        drawNetwork(layerConfig, data.weights); // Function to draw the network
    })
    .catch(error => console.error('Error:', error));
}

function drawNetwork(layerConfig, weights) {
    const canvas = document.getElementById('network-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas before redrawing

    const layerWidth = canvas.width / (layerConfig.length + 1);
    const maxNeurons = Math.max(...layerConfig);
    const neuronRadius = 20;
    const verticalSpacing = canvas.height / (maxNeurons + 1);

    function drawNeuron(x, y) {
        ctx.beginPath();
        ctx.arc(x, y, neuronRadius, 0, 2 * Math.PI);
        ctx.fillStyle = '#6fa3ef';
        ctx.fill();
        ctx.lineWidth = 4
        ctx.strokeStyle = '#000';
        ctx.stroke();
    }

    function drawConnection(x1, y1, x2, y2, weight) {
        const lineThickness = Math.abs(weight) * 5;
        ctx.lineWidth = lineThickness;

        if (weight < 0) {
            ctx.strokeStyle = 'red';
        } else {
            ctx.strokeStyle = 'green';
        }

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }

    let layerX = layerWidth;
    for (let i = 0; i < layerConfig.length; i++) {
        const numNeurons = layerConfig[i];
        const layerYStart = (canvas.height - (numNeurons - 1) * verticalSpacing) / 2;

        for (let j = 0; j < numNeurons; j++) {
            const neuronX = layerX;
            const neuronY = layerYStart + j * verticalSpacing;
            drawNeuron(neuronX, neuronY);

            if (i > 0) {
                const prevLayerNeurons = layerConfig[i - 1];
                const prevLayerYStart = (canvas.height - (prevLayerNeurons - 1) * verticalSpacing) / 2;
                for (let k = 0; k < prevLayerNeurons; k++) {
                    const prevNeuronX = layerX - layerWidth;
                    const prevNeuronY = prevLayerYStart + k * verticalSpacing;
                    const weight = weights[i - 1][j][k];  // Get the weight between the neurons
                    drawConnection(prevNeuronX, prevNeuronY, neuronX, neuronY, weight);
                }
            }
        }

        layerX += layerWidth;
    }
}
