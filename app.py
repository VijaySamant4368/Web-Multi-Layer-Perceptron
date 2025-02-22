from flask import Flask, request, jsonify, render_template
from utils import initParams

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')  # This serves the HTML page

@app.route('/get-weights', methods=['POST'])
def get_weights():
    data = request.get_json()  # Get the JSON data from the request
    layers_config = data['layers_config']  # Expected to be a list of layer configurations (neurons in each layer)
    
    # Generate random weights for the neural network based on the layer configuration
    weights, _ = initParams(layers_config)
    print(weights)
    flat_weights = [w.tolist() for w in weights]

    
    return jsonify({'weights': flat_weights})  # Return the weights as a JSON response

if __name__ == '__main__':
    app.run(debug=True)
