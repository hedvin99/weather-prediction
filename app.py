from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('weather_model.pkl')
scaler = joblib.load('scaler.pkl')

# Mapping numerical values to weather categories
weather_category_mapping = {
    0: 'Sunny',
    1: 'Rainy',
    2: 'Cloudy',
    3: 'Snowy',
    4: 'Foggy'
}

@app.route('/', methods=['GET', 'POST'])
def predict_weather():
    prediction = None
    if request.method == 'POST':
        # Get the form values
        temperature = float(request.form['Temperature'])
        dew_point = float(request.form['Dew_Point'])
        humidity = float(request.form['Humidity'])
        wind_speed = float(request.form['Wind_Speed'])
        visibility = float(request.form['Visibility'])
        pressure = float(request.form['Pressure'])
        
        # Prepare the data for prediction
        input_data = [[temperature, dew_point, humidity, wind_speed, visibility, pressure]]
        scaled_data = scaler.transform(input_data)
        
        # Make prediction using the model
        predicted_category_num = model.predict(scaled_data)[0]
        
        # Map numerical prediction to categorical label
        prediction = weather_category_mapping.get(predicted_category_num, "Unknown")

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)







