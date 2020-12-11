#!/usr/bin/env python
# coding: utf-8
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import os
import joblib
import streamlit.components.v1 as components


# @st.cache
def load_data():
    data = pd.read_csv(r"D:/Streamlit_apps/tests/D3.csv")
    return data


def main():
    st.write("First DataScience WebApp")
    st.title("Covid19 Regression with Machine Learning")
    st.subheader("Stephen Vu")

    html_temp = """
    #<div style="background-color: tomato; padding:15px">

    #<h2> My Platform </h2>
    #</div>


    """
    st.markdown(html_temp, unsafe_allow_html=True)
    data = load_data()
    st.subheader("Covid Data Set")

    # make choices
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
            st.info("General Information")
            # You can read a markdown file from supporting resources folder
            st.markdown("Some information here")

            st.subheader("Raw Twitter data and label")
            if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
                st.write(data)  # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":
            st.info("Prediction with ML Models")
            # Creating a text box for user input
            tweet_text = st.text_area("Enter Text", "Type Here")

            if st.button("Classify"):
                # Transforming user input with vectorizer
                vect_text = tweet_cv.transform([tweet_text]).toarray()
                # Load your .pkl file with the model of your choice + make predictions
                # Try loading in multiple models to give the user a choice
                predictor = joblib.load(
                    open(os.path.join("D:/Streamlit_apps/tests/DTT.pickle"), "rb"))
                prediction = predictor.predict(vect_text)

                # When model has successfully run, will print prediction
                # You can use a dictionary or similar structure to make this output
                # more human interpretable.
                st.success("Text Categorized as: {}".format(prediction))

    components.html(
        """
         
    <div class="container">
      <h2>HackerShrine</h2>
     
        <div class="card" style="width:400px">
         
        <div class="card-body ">
          <form action="/upload" method="post" enctype="multipart/form-data">
          <p class="card-text">Custom HTML </p>
            <input type="file" name="file" value="file">
            <hr>
          <input type="submit" name="upload" value="Upload" class="btn btn-success">
          </form>
         
        </div>
      </div>
      <br>
    </div>
    """,
        height=600,
    )


if __name__ == '__main__':
    main()
