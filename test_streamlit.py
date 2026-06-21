import streamlit as st

# Title
st.title("My First Streamlit App")

# Text
st.write("Hello World!")

# Input box
name = st.text_input("Enter your name:")

# Button
if st.button("Greet Me"):
    st.write(f"Hello {name}! 👋")