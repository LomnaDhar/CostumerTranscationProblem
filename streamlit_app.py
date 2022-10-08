import streamlit as st
import pandas as pd
import numpy as np
import pickle


def main():
    st.title('Customer Transaction Prediction')
    st.text("In this project I am predicting the probability that the customer will make a future transaction using the data provided by that particular customer")
    menu = ["Home","Predict"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
    if choice == "Predict":
        st.subheader("Get The Probability")

    return choice

if __name__=="__main__":
    choice = main()



if choice == "Home":
    @st.cache
    def load_data():
        data = pd.read_csv("train.csv", nrows=10)
        return data

    st.subheader('Dataset')
    st.text("The Dataset is downloaded from Kaggle's Santander Customer Transaction Prediction Competition")
    st.text("Top 10 Rows of the Data frame")
    data = load_data()
    st.dataframe(data)
    st.text("Number of Rows: 200000")
    st.text("Number of Columns: 202 (including Target and Id_code Column")


if choice == "Predict":
    @st.cache
    def load_model():
        final_model = pickle.load(open("Final_Model", "rb"))
        return final_model


    model = load_model()
    predict_file = st.file_uploader("Upload Customer Data", type=["csv"])
    if predict_file:
        file_details = {"File Name":predict_file.name, "File Type":predict_file.type, "File Size":predict_file.size}
        st.write(file_details)
        input_data = pd.read_csv(predict_file)
        st.dataframe(input_data)

        # feature engineering
        idx = input_data.columns.values
        input_data["sum"] = input_data[idx].sum(axis=1)
        input_data["min"] = input_data[idx].min(axis=1)
        input_data["max"] = input_data[idx].max(axis=1)
        input_data["mean"] = input_data[idx].mean(axis=1)
        input_data["std"] = input_data[idx].std(axis=1)
        input_data["skew"] = input_data[idx].skew(axis=1)
        input_data["kurt"] = input_data[idx].kurtosis(axis=1)
        input_data["med"] = input_data[idx].median(axis=1)
        for var in input_data.columns:
            input_data["r2__" + var] = np.round(input_data[var], 2)
            input_data["r1__" + var] = np.round(input_data[var], 1)

        # prediction
        if st.button('Submit'):
            prob = model.predict(input_data)
            st.text(f"The probability that this person will make a future transaction is {prob}")



