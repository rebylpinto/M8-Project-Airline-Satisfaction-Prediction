#Importing Libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
#######################################################

# Set page title
st.set_page_config(page_title="Airline Satisfaction Prediction Project")

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!", "About"])
# Load dataset
#cleaned train dataset
df = pd.read_csv('/Users/rebyl/Documents/Coding Temple/M8-Project Airline Satisfaction Prediction/data/clean_train.csv')
df=df.drop(columns='id')

# Home Page
if page == "Home":
    st.title("Unlocking Passenger Satisfaction: A Machine Learning Classification Study")
    st.subheader("By Rebyl Peddity")
    st.write("""
        Dive into the world of airline passenger satisfaction with this interactive Streamlit application!
    """)
    st.image('/Users/rebyl/Documents/Coding Temple/M8-Project Airline Satisfaction Prediction/flight-4516478_1280.jpg')
    

# Data Overview
elif page == "Data Overview":
    st.title("Data Overview")

    
    st.write("""
      Imagine over 100,000 passengers sharing their airline experiences. That's the power of the dataset used in this app. Each survey provides ratings on 14 distinct pre-flight and in-flight service aspects, revealing what truly matters to travelers.

We also have a wealth of passenger and flight details, blending categorical and numerical data to paint a comprehensive picture. The ultimate goal? Predicting whether a passenger's overall experience was 'satisfied' or 'neutral/dissatisfied'.

This data-driven approach allows us to explore the nuances of passenger satisfaction and build predictive models for actionable airline improvements.
    """)
    st.subheader("Data Dictionary")
    st.markdown("""
    | Feature   | Explaination|
    | ----------- | ------------ |  
    | *Gender* | male/female |        
    | *Customer Type* | loyal/disloyal | 
    | *Age* | Age of the customer |      
    |*Type of Travel*| business/personal|          
    |*Class*| eco/eco plus/business|           
    |*Flight Distance*| distance in miles|          
    |*Inflight WiFi Service*| Satisfaction rating of inflight Wi-Fi service - 0:Not Applicable, Rating: 1-5|           
    |*Departure/Arrival Time Convenience*| Convenience of departure/arrival time - 0:Not Applicable, Rating: 1-5|          
    |*Ease of Online Booking*| Satisfaction rating of booking online - 0:Not Applicable, Rating: 1-5|            
    |*Gate Location*| Satisfaction rating of gate location - 0:Not Applicable, Rating: 1-5|            
    |*Food and Drink*| Satisfaction rating of inflight food and drink - 0:Not Applicable, Rating: 1-5|         
    |*Online Boarding*| Satisfaction rating of  online boarding process - 0:Not Applicable, Rating: 1-5|          
    |*Seat Comfort*| Satisfaction rating of comfort of seat - 0:Not Applicable, Rating: 1-5|          
    |*Inflight Entertainment*| Satisfaction rating of inflight entertainment - 0:Not Applicable, Rating: 1-5|     
    |*On-board Service*| Satisfaction rating of on-board service - 0:Not Applicable, Rating: 1-5|          
    |*Leg Room Service*| Satisfaction rating of  legroom - 0:Not Applicable, Rating: 1-5|          
    |*Baggage Handling*| Satisfaction rating of  baggage handling - 0:Not Applicable, Rating: 1-5|         
    |*Check-in Service*| Satisfaction rating of check-in service - 0:Not Applicable, Rating: 1-5|      
    |*Inflight Service*| Satisfaction rating of inflight service - 0:Not Applicable, Rating: 1-5|         
    |*Cleanliness*|  Satisfaction rating of cleanliness - 0:Not Applicable, Rating: 1-5|         
    |*Departure Delay in Minutes*| departure delay|         
    |*Arrival Delay in Minutes*| arrival delay|        
    |*Satisfaction*| satisfied/neutral or dissatisfied|
    """)

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis\nUsing Plotly Visualizations")

    container = st.container(border=True)
    container.subheader("Select the type of visualization you'd like to explore:")
    eda_type = container.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Correlation Heatmap'], default=['Correlation Heatmap'], key="eda_type_multiselect")
    
    num_cols = df.select_dtypes(include='number').columns.tolist()

    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox("Select a numeric variable for the histogram:", num_cols, key="histogram_selectbox")
        if h_selected_col:
            chart_title = f"Distribution of {h_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating", key="histogram_satisfaction_checkbox"):
                st.plotly_chart(px.histogram(df, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df, x=h_selected_col, title=chart_title))

    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox("Select a numeric variable for the box plot:", num_cols, key="boxplot_selectbox")
        if b_selected_col:
            chart_title = f"Distribution of {b_selected_col.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating", key="boxplot_satisfaction_checkbox"):
                st.plotly_chart(px.box(df, x=b_selected_col, color='satisfaction', title=chart_title))
            else:
                st.plotly_chart(px.box(df, x=b_selected_col, title=chart_title))

    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox("Select x-axis variable:", num_cols, key="scatter_x_selectbox")
        selected_col_y = st.selectbox("Select y-axis variable:", num_cols, key="scatter_y_selectbox")
        if selected_col_x and selected_col_y:
            chart_title = f"{selected_col_x.title().replace('_', ' ')} vs. {selected_col_y.title().replace('_', ' ')}"
            if st.checkbox("Show by Overall Satisfaction Rating", key="scatterplot_satisfaction_checkbox"):
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))
            else:
                st.plotly_chart(px.scatter(df, x=selected_col_x, y=selected_col_y, title=chart_title))
 
    if 'Correlation Heatmap' in eda_type:
        st.subheader("Correlation of Features with Satisfaction")
        try:
            correlation_matrix = df.corr(numeric_only=True)[['satisfaction']].sort_values(by='satisfaction', ascending=False)
            plt.figure(figsize=(8, 10))
            sns.heatmap(correlation_matrix, vmin=-1, vmax=1, annot=True, cmap='coolwarm')
            st.pyplot(plt)
        except Exception as e:
            st.error(f"An error occurred while generating the heatmap: {e}")


elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")
########
########

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

elif page == "Make Predictions!":
    st.title("Make Your Own Airline Passenger Satisfaction Prediction")
    container = st.container(border=True)
    container.subheader("Use 22 features to input in a Random Forest classification model")
    container.subheader("**Adjust the feature scale values below to make your own predictions on whether an airpline passenger will be satisfied or not**")
    

    # User inputs for prediction
    gender = st.slider("Gender ---> Female: 0, Male: 1", min_value=0, max_value=1, value=1)
    customer_type = st.slider("Customer Type ---> Disloyal Customer: 0, Loyal Customer: 1", min_value=0, max_value=1, value=1)
    age = st.slider("Age --> Pick An Age From 1 To 90", min_value=1, max_value=90, value=50)
    travel_type = st.slider("Type of Travel ---> Personal Travel: 0 , Business Travel: 1", min_value=0, max_value=1, value=1)
    travel_class = st.slider("Travel Class Types ---> Eco: 1, Eco Plus: 2, Business: 3", min_value=1, max_value=3, value=2)
    flight_distance = st.slider("Flight Distance ---> Pick A Distance Between 1 and 5000 Miles", min_value=1, max_value=5000, value= 2500)
    inflight_wifi = st.slider("Inflight WiFi Service Rating ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    time_convenient = st.slider("Departure/Arrival Time Convenience ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    online_booking = st.slider("Ease of Online Booking ---> Pick A Rating Between 1 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    gate_location = st.slider("Gate Location ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=0)
    food_drink = st.slider("Food and Drink ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=2)
    online_boarding = st.slider("Online Boarding ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    seat_comfort = st.slider("Seat Comfort ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=1)
    entertainment = st.slider("Inflight Entertainment---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    on_board_service = st.slider("On-Board Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=2)
    leg_room = st.slider("Leg Room Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    baggage_handling = st.slider("Baggage Handling ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=5)
    checkin_service = st.slider("Check-in Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=0)
    inflight_service = st.slider("Inflight Service ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    cleanliness = st.slider("Cleanliness ---> Pick A Rating Between 0 and 5, If Not Applicable, Choose 0:", min_value=0, max_value=5, value=3)
    departure_delay = st.slider("Departure Delay ---> Pick A Rating Between 0 and 1600 Minutes", min_value=0, max_value=1600, value=60)
    arrival_delay = st.slider("Arrival Delay ---> Pick A Rating Between 0 and 1200 Minutes", min_value=0, max_value=1200, value=60)

    # User input dataframe
    user_input = pd.DataFrame({
        'gender': [gender],
        'customer type': [customer_type],
        'age': [age],
        'type of travel': [travel_type],
        'class': [travel_class],
        'flight distance': [flight_distance],
        'inflight wifi service': [inflight_wifi],
        'departure/arrival time convenient': [time_convenient],
        'ease of online booking': [online_booking],
        'gate location': [gate_location],
        'food and drink': [food_drink],
        'online boarding': [online_boarding],
        'seat comfort': [seat_comfort],
        'inflight entertainment': [entertainment],
        'on-board service': [on_board_service],
        'leg room service': [leg_room],
        'baggage handling': [baggage_handling],
        'checkin service': [checkin_service],
        'inflight service': [inflight_service],
        'cleanliness': [cleanliness],
        'departure delay in minutes': [departure_delay],
        'arrival delay in minutes': [arrival_delay]
    })

    st.write("### Your Input Values:")
    st.dataframe(user_input)

    # Using Random Forest model for predictions since this was the most accurate in terms of understanding the training and test data:
    model = RandomForestClassifier()
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction = model.predict(user_input_scaled)[0]


    # Display the result
    st.write(" ### Based on your input features, the model predicts that particular airline passenger will be:")
    st.write(f"# {prediction}")

elif page == "About":
    st.title("About Airline Satisfaction Prediction")

    st.write(
        """
        This application utilizes a machine learning classifier to predict customer satisfaction based on various factors related to their airline travel experience. 
        The model analyzes features such as flight distance, departure and arrival delays, seat comfort, in-flight service, and other relevant metrics to determine 
        whether a customer is likely to be satisfied or dissatisfied.
        """
    )

    st.subheader("Purpose:")
    st.write(
        """
        The primary goal of this model is to provide insights into the key drivers of customer satisfaction within the airline industry. 
        By accurately predicting satisfaction levels, airlines can:
        """
    )
    st.markdown(
        """
        * **Identify areas for improvement:** Pinpoint specific aspects of their service that contribute to customer dissatisfaction.
        * **Enhance customer experience:** Implement targeted strategies to address identified issues and improve overall satisfaction.
        * **Optimize resource allocation:** Focus resources on areas with the greatest impact on customer satisfaction.
        * **Proactively manage customer relations:** Anticipate potential dissatisfaction and take preventative measures.
        """
    )

    st.subheader("How it Works:")
    st.write(
        """
        The model is trained on a dataset containing historical airline customer feedback and corresponding satisfaction ratings. 
        It leverages machine learning algorithms to learn the relationships between the input features and the target variable (satisfaction). 
        Once trained, the model can predict the satisfaction level for new, unseen customer data.
        """
    )

    st.subheader("Key Features Analyzed:")
    st.write(
        """
        The model considers a range of features, including but not limited to:
        """
    )
    st.markdown(
        """
        * **Flight Information:** Distance, delays, etc.
        * **Service Quality:** Seat comfort, in-flight entertainment, food and drink, etc.
        * **Customer Service:** Online boarding, check-in, baggage handling, etc.
        """
    )

    st.subheader("Intended Audience:")
    st.write(
        """
        This application is designed for airline professionals, customer service managers, and data analysts who seek to understand and improve customer satisfaction within the aviation industry.
        """
    )

    st.subheader("Disclaimer:")
    st.write(
        """
        The predictions generated by this model are based on the data it was trained on and should be used as a tool for analysis and decision-making. 
        While the model aims for accuracy, it's important to consider that real-world customer satisfaction can be influenced by various factors not captured in the training data.
        """
    )    