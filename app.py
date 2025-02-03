from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import joblib
from geopy.distance import great_circle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key' 


model = joblib.load(r'C:\Users\hyper\OneDrive\Desktop\DevRep\frontend\best_model.pkl')


data = pd.read_csv('synthetic_fish_population_cleaned_with_species2.csv')


users = {
    'testuser': {'password': 'password123', 'field_of_work': 'fisherman'},
    'admin': {'password': 'adminpass', 'field_of_work': 'researcher'}
}

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    user = get_current_user()
    if user is None:
        return redirect(url_for('login'))

    field_of_work = user['field_of_work']

    if request.method == 'POST':
        if field_of_work == 'fisherman':
            try:
                latitude = float(request.form['latitude'])
                longitude = float(request.form['longitude'])
                temperature = float(request.form['temperature'])

                new_data = pd.DataFrame({
                    'Latitude': [latitude],
                    'Longitude': [longitude],
                    'Temperature': [temperature],
                })
                new_data.to_csv('synthetic_fish_population_cleaned_with_species2.csv', mode='a', header=False, index=False)

                flash('Data added successfully!', 'success')
                return redirect(url_for('index'))
            except ValueError:
                flash('Invalid input. Please enter numeric values.', 'error')

        else:
            try:
                latitude = float(request.form['latitude'])
                longitude = float(request.form['longitude'])
                temperature = float(request.form['temperature'])
                sst = float(request.form['sst'])
                salinity = float(request.form['salinity'])
                river_discharge = float(request.form['river_discharge'])
                wind_speed = float(request.form['wind_speed'])
                precipitation = float(request.form['precipitation'])

                new_data = pd.DataFrame({
                    'Latitude': [latitude],
                    'Longitude': [longitude],
                    'Temperature': [temperature],
                    'SST': [sst],
                    'Salinity': [salinity],
                    'River_Discharge': [river_discharge],
                    'Wind_Speed': [wind_speed],
                    'Precipitation': [precipitation]
                })
                new_data.to_csv('synthetic_fish_population_cleaned_with_species2.csv', mode='a', header=False, index=False)

                flash('Data added successfully!', 'success')
                return redirect(url_for('index'))
            except ValueError:
                flash('Invalid input. Please enter numeric values.', 'error')

    if field_of_work == 'fisherman':
        @app.route('/add_data_fisherman', methods=['GET', 'POST'])
        def add_data_fisherman():
            if request.method == 'POST':
                latitude = request.form.get('latitude')
                longitude = request.form.get('longitude')
                temperature = request.form.get('temperature')
                # Handle the data, save it to the database, etc.
                flash('Data added successfully!', 'success')
                return redirect(url_for('dashboard'))
                return render_template('add_data_fisherman.html')
            else:
                return render_template('add_data_other.html')
    
@app.route('/add_data_fisherman', methods=['GET', 'POST'])
def add_data_fisherman():
    if request.method == 'POST':
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        temperature = request.form.get('temperature')
      
        flash('Data added successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('add_data_fisherman.html')




@app.route('/add_data_other', methods=['GET', 'POST'])
def add_data_other():
    if request.method == 'POST':
        # Process form data here
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        temperature = request.form.get('temperature')
        date = request.form.get('date')
        additional_info = request.form.get('additional_info')
       
        flash('Data added successfully!', 'success')
        return redirect(url_for('dashboard')) 
    return render_template('add_data_other.html')





@app.route('/fish_catch', methods=['GET', 'POST'])
def fish_catch():
    return redirect(url_for('predict'))

@app.route('/show_data')
def show_data():

    data = pd.read_csv('synthetic_fish_population_cleaned_with_species2.csv')

    data_html = data.to_html(classes='table table-striped', index=False)
    return render_template('show_data.html', data_html=data_html)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username]['password'] == password:
            session['username'] = username  # Save the username in the session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        field_of_work = request.form['field_of_work']
        
        if username in users:
            flash('Username already exists. Please choose another one.', 'error')
        else:
            users[username] = {'password': password, 'field_of_work': field_of_work}
            flash('Registration successful! You can now login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            port_lat = float(request.form['port_lat'])
            port_lon = float(request.form['port_lon'])
            fish_species = request.form['fish_species']

            target_coords = (port_lat, port_lon)

            features = ['SST', 'Salinity', 'River_Discharge', 'Wind_Speed', 'Precipitation', 'Latitude', 'Longitude']
            predictions_df = data[features].copy()

            predictions_df['Predicted_Fish_Population'] = model.predict(predictions_df[features])

            predictions_df['Latitude'] = data['Latitude']
            predictions_df['Longitude'] = data['Longitude']

            nearest_location = search_species_nearme(target_coords, predictions_df, fish_species)

            if nearest_location is None:
                return 'No nearby population found for the selected species.'

            distance = great_circle(target_coords, (nearest_location['Latitude'], nearest_location['Longitude'])).kilometers

            return render_template('result.html', nearest_location=nearest_location, distance=distance, fish_species=fish_species)
        
        except Exception as e:
            return f"An error occurred: {e}"

    return render_template('predict.html')

def search_species_nearme(target_coords, predictions_df, fish_species, population_threshold=0.5):
    species_df = data[data['Fish_Species'] == fish_species]

    if species_df.empty:
        return None

    try:
        merged_df = pd.merge(predictions_df, species_df, on=['Latitude', 'Longitude'])
    except KeyError as e:
        return f"Merge error: {e}"

    if 'Predicted_Fish_Population' not in merged_df.columns:
        return None

    predictions_species_df = merged_df[merged_df['Predicted_Fish_Population'] >= population_threshold]

    if predictions_species_df.empty:
        return None

    predictions_species_df['Distance_to_Target'] = predictions_species_df.apply(
        lambda row: great_circle(target_coords, (row['Latitude'], row['Longitude'])).kilometers, axis=1
    )

    nearest_location = predictions_species_df.loc[predictions_species_df['Distance_to_Target'].idxmin()]
    return nearest_location

def get_current_user():
    username = session.get('username')  # Retrieve the username from the session
    return users.get(username)

if __name__ == '__main__':
    app.run(debug=True)
