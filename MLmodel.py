import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point



import pandas as pd
import numpy as np

def create_synthetic_data():
    num_samples = 10
    
  
    latitude_min = 5.0  
    latitude_max = 37.0  
    longitude_min = 68.0  
    longitude_max = 92.0  
    

    import numpy as np
    latitude_min = 8
    latitude_max = 25
    longitude_min = 68
    longitude_max = 90
    num_samples = 100000
    data = {
    'Latitude': np.round(np.random.uniform(low=latitude_min, high=latitude_max, size=num_samples), 3),
    'Longitude': np.round(np.random.uniform(low=longitude_min, high=longitude_max, size=num_samples), 3),
    'SST': np.round(np.random.uniform(low=22, high=32, size=num_samples), 3),                                         # Sea surface temperature in Celsius
    'Salinity': np.round(np.random.uniform(low=30, high=37, size=num_samples), 3),                                    # Salinity in PSU (Practical Salinity Units)
    'River_Discharge': np.round(np.random.uniform(low=100, high=3000, size=num_samples), 3),                          # River discharge (m3/s)
    'Wind_Speed': np.round(np.random.uniform(low=0, high=15, size=num_samples), 3),                                   # Wind speed in m/s
    'Precipitation': np.round(np.random.uniform(low=0, high=500, size=num_samples), 3),                               # Precipitation in mm
    'Fish_Population': np.round(np.random.uniform(low=100, high=5000, size=num_samples), 3),                          # Synthetic fish population (CPUE or abundance)
    'SSH': np.round(np.random.uniform(low=0, high=60, size=num_samples), 3),                                          # Sea Surface Height in meters
    'Current_Speed': np.round(np.random.uniform(low=0, high=2, size=num_samples), 3)                                  # fCurrent speed in m/s
}

    df = pd.DataFrame(data)
    df.to_csv('synthetic_fish_population.csv', index=False)
    
    def remove_outliers(df, column_name):
      
        z_scores = (df[column_name] - df[column_name].mean()) / df[column_name].std()
      
        threshold = 3
        # Filter out outliers
        return df[np.abs(z_scores) < threshold]
    
    for column in ['SST', 'Salinity', 'River_Discharge', 'Wind_Speed', 'Precipitation', 'Latitude', 'Longitude','SSH','Current_Speed']:
        df = remove_outliers(df, column)
        df.to_csv('synthetic_fish_population.csv', index=False)

create_synthetic_data()


#--------------------------------------------------------------------------------------------------------------------------------------------#
create_synthetic_data()

data=pd.read_csv('synthetic_fish_population.csv')
print(data.head())

features=['SST', 'Salinity', 'River_Discharge', 'Wind_Speed', 'Precipitation', 'Latitude', 'Longitude','SSH','Current_Speed']
X = data[features]
y = data['Fish_Population']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importance in Fish Population Prediction')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Fish Population')
plt.ylabel('Predicted Fish Population')
plt.title('Actual vs Predicted Fish Population')
plt.show()
