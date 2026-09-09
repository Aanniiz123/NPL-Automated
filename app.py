import streamlit as st

# Set the page title
st.title("Hello World! 👋")

# Add some descriptive text
st.write("This is a simple template to get your Streamlit app running.")

# Add an interactive text input widget
user_name = st.text_input("What is your name?", placeholder="Type your name here...")

# Add a button that triggers an action when clicked
if st.button("Celebrate!"):
    if user_name:
        st.success(f"Welcome aboard, {user_name}! 🎉")
        st.balloons()  # Fun animation
    else:
        st.warning("Please enter a name first! ⚠️")
