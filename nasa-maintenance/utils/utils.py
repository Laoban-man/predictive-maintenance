import streamlit as st
import graphviz as graphviz

def print_data_analysis():
    st.title("Data Analysis")
    st.info("Data analysis inspects the contents of the data and provides statistics describing them.")

def print_data():
	st.markdown("The dataset is a collection of time series from various sensor outputs from multiple engines:\
                - 4 training sets,\
                - 4 testing sets,\
                - 4 Remaining Useful Life (RUL) in cycles of the testing data.")

	st.markdown("reference : https://www.kaggle.com/c/predictive-maintenance", unsafe_allow_html=True)

def print_intro():
	st.title("NASA engine maintenance prediction")
	st.info("This app is created as a simple implementation of streamlit in the context of machine learning for a time series.")

def print_outline():
	st.title("Outline")
	# GOAL
	st.subheader("Goal:")
	st.markdown("Build a application providing EDA, data preparation, and data modeling to predict leftover cycles before engine maintenance.")
	# DATA
	st.subheader("Data:")
	st.markdown("Data collection was a simple matter of downloading via the Kaggle API and then proceeding with basic ")

	# Predictions
	st.subheader("Predictions:")
	st.markdown("Based on the model selected, predictions are provided on training validation data and training sets.")
