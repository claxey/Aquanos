import streamlit as st
import pandas as pd

def main():
    st.set_page_config(page_title='Fish Population Prediction')
    
    # Input for Latitude and Longitude
    latitude = st.number_input('Enter Latitude', min_value=-90.0, max_value=90.0, value=0.0)
    longitude = st.number_input('Enter Longitude', min_value=-180.0, max_value=180.0, value=0.0)
    
    # Dropdown for Fish Species
    species_options = ['Hilsa', 'Pomfret', 'Sardine', 'Tuna', 'Mackerel']
    selected_species = st.selectbox('Select Fish Species', species_options)
    
    # Display selected information
    st.write(f"Latitude: {latitude}, Longitude: {longitude}, Selected Species: {selected_species}")

    # Options for further actions
    options = ['Add Data', 'Predict Data', 'Check Dataset']
    selected_option = st.selectbox('Choose an option', options)
    
    if selected_option == 'Add Data':
        st.write('Add data functionality goes here')
    elif selected_option == 'Predict Data':
        st.write('Predict data functionality goes here')
    elif selected_option == 'Check Dataset':
        st.write('Check dataset functionality goes here')

if __name__ == '__main__':
    main()