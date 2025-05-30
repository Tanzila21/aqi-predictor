# Ensure streamlit is installed before running this file

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestRegressor 

# Load your dataset
def load_data():
    df = pd.read_csv("updated_pollution_dataset.csv")  # Replace with your actual dataset
    return df

def load_model():
    with open("aqi_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def main():
    st.title("Air Quality Index (AQI) Prediction")
    
    # Sidebar navigation
    menu = ["Introduction", "EDA", "Model", "Conclusion"]
    choice = st.sidebar.selectbox("Navigation", menu)
    
    df = load_data()
    model = load_model()
    
    if choice == "Introduction":
        st.subheader("Introduction")
        st.write("This project aims to predict AQI using machine learning techniques.")
        st.write("Dataset contains various air pollutant levels and air quality categories.")
        st.write("0: Good")
        st.write("1: Hazardous")
        st.write("2: Moderate")
        st.write("3: Poor")

    
    elif choice == "EDA":
        st.subheader("Exploratory Data Analysis")
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        
        st.write("### Statistical Summary")
        st.write(df.describe())

        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10,6))
        numeric_df = df.select_dtypes(include=['number'])  # Keep only numeric columns
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.show()
        st.pyplot(plt)
        

        # Allow user to select a numeric column for visualization
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.error("No numeric columns found in the dataset!")
            return

        selected_column = st.selectbox("Select a column for visualization", numeric_cols)

        # ---- HISTOGRAM ----
        st.subheader(f"Histogram of {selected_column}")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column], kde=True, bins=30, ax=ax)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # ---- BOX PLOT ----
        st.subheader(f"Box Plot of {selected_column}")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[selected_column], ax=ax)
        ax.set_ylabel(selected_column)
        st.pyplot(fig)
 
        # ---- SCATTER PLOT ----
        st.subheader("Scatter Plot")
        col_x = st.selectbox("Select X-axis variable", numeric_cols, index=0)
        col_y = st.selectbox("Select Y-axis variable", numeric_cols, index=1)

        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col_x], y=df[col_y], ax=ax)
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        st.pyplot(fig)

    
    elif choice == "Model":
        st.subheader("Model Prediction")
        st.write("Enter the pollutant levels to predict AQI:")
        
        # Collecting user input
        input_features = {}
        for col in df.columns[:-1]:  # Assuming last column is AQI
            input_features[col] = st.number_input(f"{col}", value=float(df[col].mean()))
        
        if st.button("Predict AQI"):
            input_df = pd.DataFrame([input_features])
            prediction = model.predict(input_df)
            st.success(f"Predicted AQI: {prediction[0]:.2f}")
    
    elif choice == "Conclusion":

        st.subheader("Conclusion")

        st.write("The AQI prediction model helps in understanding air pollution levels.")

        st.write("Further improvements can be made using deep learning techniques.")

        st.write("This project focused on developing a predictive model for Air Quality Index (AQI) Categories using machine learning techniques. The key takeaways from this project are:")
      
        st.write("Data Preprocessing: The quality of data significantly impacted model accuracy. Handling missing values, scaling features, and encoding categorical variables were crucial steps in preparing the data for modeling.")
    
        st.write("Model Selection: Two machine learning algorithms were tested, including  decision trees and random forests. Random Forests performed the best due to their ability to handle complex relationships in the data and provide accurate predictions.")

        st.write("Model Evaluation: The model was evaluated using metrics such as RMSE and R^2, with the final model achieving a high level of accuracy in predicting AQI levels based on environmental factors.")

        st.write("Feature Importance: Analysis of feature importance revealed that certain variables, such as CO, temperature and proximity to industrial areas were the most influential in determining AQI levels.")

        st.write("Real-World Application: The model can be used to forecast AQI categories, which is critical for public health and safety. With further refinement and real-time data integration, this system can be adapted for use in smart cities or environmental monitoring systems.")

        st.write("Overall, the project demonstrated the effectiveness of machine learning models in environmental prediction tasks and highlighted the potential for further optimization and expansion.")

if __name__ == "__main__":
    main()
