import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import google.generativeai as genai
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the dataset with a specified encoding
data = pd.read_csv('Food Drive Data Collection 2024(1-448).csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
   st.image('main image.png', caption="Edmonton Food Drive", use_container_width=True)

      # Add logos on the right (col3)

      # Highlight the project title
   st.title("Edmonton Food Drive Dashboard")

    # Add three summary statistics cards
   st.write("### Project Highlights")
   col1, col2, col3 = st.columns(3)
   col1.metric(label="Donation Bags (2024)", value="14,649")
   col2.metric(label="Volunteers", value="1600+")
   col3.metric(label="Neighborhoods Covered", value="50+")

    # Use tabs to organize content
   tab1, tab2, tab3 = st.tabs(["Abstract", "What the Project Does", "Inspiration"])

    # Abstract Tab
   with tab1:
        st.write("💡 **Abstract**")
        st.info(
            """
            This project focuses on optimizing the logistics and predicting the success of donation drives
            for the Edmonton Food Drive initiative. Using data from 2023 and 2024, the project analyzes patterns
            in donation bag counts and volunteer contributions across Edmonton's neighborhoods.
            """
        )

    # What the Project Does Tab
   with tab2:
        st.write("👨🏻‍💻 **What Our Project Does**")
        bullet_points = [
            "Predicts donation bag collection using machine learning.",
            "Identifies key factors for successful donation drives.",
            "Optimizes routes and volunteer distribution for efficiency.",
            "Visualizes trends and patterns in donation collection.",
            "Provides actionable insights to support decision-making.",
        ]
        for point in bullet_points:
            st.markdown(f"- {point}")

    # Inspiration Tab
   with tab3:
        st.write("🌟 **Inspiration**")
        st.warning(
            """
            The Edmonton Food Drive Project demonstrates how technology and community collaboration
            can address food insecurity and maximize the impact of donation drives. By analyzing
            patterns and optimizing logistics, it ensures efficient resource allocation and fosters
            societal good.
            """
        )

def data_visualizations(data):
    st.title("Data Visualizations")

    st.write("### Interactive Visualizations and Insights")

    # Show metrics at the top


    try:
        # Dropped the irrelevant columns
        data = data.drop(
            ['Name', 'Email', 'ID', 'Start time', 'Completion time',
             'Other Drop-off Locations', 'Sherwood Park Stake', 'Email address '],
            axis=1
        )

        # Drop row 1
        data = data.drop(1)

        # Combine the specified columns into a new column 'Stake'
        data['Ward'] = (
            data['Bonnie Doon Stake'].fillna('') +
            data['Edmonton North Stake'].fillna('') +
            data['Gateway Stake'].fillna('') +
            data['Riverbend Stake'].fillna('') +
            data['YSA Stake'].fillna('')
        )

        # Drop the individual stake columns
        data = data.drop(
            ['Bonnie Doon Stake', 'Edmonton North Stake',
             'Gateway Stake', 'Riverbend Stake', 'YSA Stake'],
            axis=1
        )

        # Rename columns for clarity
        data = data.rename(columns={
            '# of Adult Volunteers who participated in this route': 'No of Adult Volunteers',
            '# of Youth Volunteers who participated in this route\n': 'No of Youth Volunteers',
            '# of Donation Bags Collected': 'Donation Bags Collected',
            'Time Spent Collecting Donations': 'Time to Complete (min)',
            'Did you complete more than 1 route?': 'Completed More Than One Route',
            '# of Doors in Route': 'Doors in Route'
        })

        # Outlier Handling
        data.loc[0, 'Donation Bags Collected'] = 59
        data.loc[59, 'No of Youth Volunteers'] = 0
        data['Drop Off Location'] = data['Drop Off Location'].str.strip()

        gateway_doors_average = data[data['Drop Off Location'] == 'Gateway Stake Centre']['Doors in Route'].mean()
        data.loc[127, 'Doors in Route'] = 195

        st.write("#### Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Donation Bags", data["Donation Bags Collected"].sum())
        col2.metric("Unique Wards", data["Ward"].nunique())
        col3.metric("Average Volunteers per Route", round(data[["No of Adult Volunteers", "No of Youth Volunteers"]].mean().mean(), 2))

        # Sidebar filters
        st.sidebar.header("Filter Data")
        selected_stake = st.sidebar.selectbox("Select Stake", data["Stake"].unique())

        filtered_data = data[(data["Stake"] == selected_stake)]

        # Visualizations
        st.write(f"#### Donation Bags Collected in {selected_stake}")
        fig = px.bar(
            filtered_data,
            x="Ward",
            y="Donation Bags Collected",
            color="Ward",
            title=f"Donation Bags by Ward in {selected_stake})",
            labels={"Ward": "Ward", "Donation Bags Collected": "Bags Collected"},
            template="plotly_white"
        )
        st.plotly_chart(fig)

        st.write(f"#### Volunteers Distribution in {selected_stake}")
        fig = px.box(
            filtered_data,
            x="Ward",
            y="No of Adult Volunteers",
            color="Ward",
            title="Distribution of Adult Volunteers",
            labels={"Ward": "Ward", "No of Adult Volunteers": "Adult Volunteers"},
            template="plotly_white"
        )
        st.plotly_chart(fig)

        st.write(f"#### Time Spent Collecting Donations in {selected_stake}")
        fig = px.histogram(
            filtered_data,
            x="Time to Complete (min)",
            nbins=20,
            title="Time Spent Distribution",
            labels={"Time to Complete (min)": "Time to Complete (minutes)"},
            template="plotly_white"
        )
        st.plotly_chart(fig)


    except Exception as e:
        st.error(f"An error occurred: {e}")




# Page 3: Machine Learning Modeling

from sklearn.preprocessing import OneHotEncoder
import numpy as np

def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict the number of donation bags collected:")

    try:
        # One-hot encode the Stake column
        encoder = OneHotEncoder(sparse_output=False)
        stake_data = encoder.fit_transform(data[['Stake']])  # Replace 'Stake' with the actual column name
        stake_classes = encoder.categories_[0]

        # Create a dropdown for Stake selection
        selected_stake = st.selectbox("Select Stake", options=stake_classes)

        # Input fields for user to enter other details
        completed_routes = st.slider("Completed More Than One Route (0 = No, 1 = Yes)", 0, 1, 0)
        time_spent = st.slider("Time to Complete (minutes)", 10, 300, 60)
        adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
        doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
        youth_volunteers = st.slider("Number of Youth Volunteers", 0, 50, 10)

        if st.button("Predict"):
            # Encode the selected Stake
            encoded_stake = np.zeros(len(stake_classes))
            encoded_stake[list(stake_classes).index(selected_stake)] = 1

            # Prepare input for prediction
            user_input = np.concatenate(([completed_routes, time_spent, adult_volunteers, doors_in_route, youth_volunteers], encoded_stake))
            user_input_df = pd.DataFrame([user_input], columns=[
                "Completed More Than One Route", "Time to Complete", "No of Adult Volunteers",
                "Doors in Route", "No of Youth Volunteers", *[f"Stake_{cls}" for cls in stake_classes]
            ])

            try:
                # Load the trained regression model
                model = joblib.load('knn_regressor_model.pkl')
                prediction = model.predict(user_input_df)
                st.success(f"Predicted Number of Donation Bags: {round(prediction[0])}")
            except FileNotFoundError:
                st.error("The regression model file is not found.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

        # You can add additional information or actions based on the prediction if needed
# Page 4: Neighbourhood Mapping
# Read geospatial data

geodata = pd.read_csv("location.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Drop rows with missing Latitude or Longitude
    geodata_clean = geodata.dropna(subset=['Latitude', 'Longitude'])

    # Ensure Neighbourhood column is cleaned
    geodata_clean['Neighbourhood'] = geodata_clean['Neighbourhood'].fillna('Unknown').str.strip().str.lower()

    # Aggregate total donation bags collected by neighborhood
    neighborhood_summary = geodata_clean.groupby('Neighbourhood', as_index=False).agg({
        'Donation Bags Collected': 'sum',
        'Latitude': 'first',  # Use the first latitude for the map
        'Longitude': 'first'  # Use the first longitude for the map
    })


    # Create a dropdown menu for neighborhood selection
    st.write("### Select a Neighborhood")
    selected_neighbourhood = st.selectbox(
        "Choose a neighborhood to view details:",
        options=neighborhood_summary['Neighbourhood'].unique()
    )

    # Filter the data based on the selected neighborhood
    filtered_data = neighborhood_summary[neighborhood_summary['Neighbourhood'] == selected_neighbourhood]

    # Display information about the selected neighborhood
    if not filtered_data.empty:
        total_bags_collected = filtered_data['Donation Bags Collected'].sum()
        st.write(f"### Total Donation Bags Collected: {total_bags_collected}")

        # Create a map for the selected neighborhood
        fig = px.scatter_mapbox(
            filtered_data,
            lat='Latitude',
            lon='Longitude',
            hover_name='Neighbourhood',
            hover_data={
                "Donation Bags Collected": True  # Show total donation bags in hover
            },
            zoom=12,
            title=f"Locations in {selected_neighbourhood.title()}"
        )

        # Update map layout to use OpenStreetMap style
        fig.update_layout(mapbox_style='open-street-map')

        # Display the map
        st.plotly_chart(fig)
    else:
        st.write("No data found for the selected neighborhood.")



# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.office.com/Pages/ResponsePage.aspx?id=8RGgKwr180SiANs-p04pt4KtM4f1RPBLgaOWpT8hVzBUMlNQS1lGOFBSQVVJUE85WUtBV0s1NFJBNi4u"
    st.markdown(f"[Fill out the form]({google_form_url})")

def thank_you_page():
    st.title("Thank You to Our Stakeholders!")

    # Add a thank you message
    st.write(
        """
        We extend our heartfelt gratitude to our amazing stakeholders for their invaluable support in making the Edmonton Food Drive a success.
        Your contributions and dedication have made a meaningful impact on our community.
        """
    )

    # Create a layout with columns for displaying logos
    st.write("### Our Stakeholders:")
    st.image('Norquest logo.png', caption="NorQuest College", use_container_width=True)
    st.image('Screenshot (18).png', caption="The Church of Jesus Christ of Latter-Day Saints", use_container_width=True)
    st.image('edmonton_logo.png', caption="City of Edmonton", use_container_width=True)
    st.image('food_bank_logo.png', caption="Feed Edmonton's Food Bank", use_container_width=True)
    st.image('just_serve_logo.png', caption="JustServe", use_container_width=True)

    # Add a footer with a final thank you message
    st.write("---")
    st.subheader("Together, we are making a difference!")



# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Data Visualizations", "ML Modeling", "Neighbourhood Mapping", "Data Collection","Thank You"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Data Visualizations":
        data_visualizations(data)
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Thank You":
        thank_you_page()


if __name__ == "__main__":
    main()
