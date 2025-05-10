import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Cart Abandonment in E-commerce",
    page_icon="🛒",
    layout="wide"
)

# Model utilities directly in this file
class CartAbandonmentModel:
    def __init__(self, model_path=None):
        """
        Initialize the cart abandonment prediction model
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model from disk"""
        try:
            self.model = joblib.load(model_path)
            st.success(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        """
        Make a prediction using the loaded model
        
        Args:
            features: Dictionary or DataFrame containing the features
            
        Returns:
            bool: True for purchase (not abandoning), False for abandonment
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # If input is a dictionary, convert to DataFrame
        if isinstance(features, dict):
            features = pd.DataFrame([features])
            
        # Make prediction
        prediction = self.model.predict(features)[0]
        return bool(prediction)

# Find the model file
def find_model_file():
    """Search for the model file in common locations"""
    # Paths to check
    possible_locations = [
        "best_model.pkl",
        "ml_models/best_model.pkl",
        "../ml_models/best_model.pkl",
        "../../ml_models/best_model.pkl",
        os.path.join(os.getcwd(), "best_model.pkl"),
        os.path.join(os.getcwd(), "ml_models/best_model.pkl"),
    ]
    
    # Show location being searched
    with st.sidebar.expander("Debug: Model Search Paths"):
        st.write("Current working directory:", os.getcwd())
        for loc in possible_locations:
            exists = os.path.exists(loc)
            st.write(f"{loc}: {'✅ Found' if exists else '❌ Not found'}")
            if exists:
                return loc
    
    # If not found automatically, let user upload model
    return None

# Load the model
@st.cache_resource
def get_model(model_path=None):
    if model_path:
        return CartAbandonmentModel(model_path)
    
    # Try to find the model
    found_path = find_model_file()
    if found_path:
        return CartAbandonmentModel(found_path)
    
    # If model not found, return empty model
    return CartAbandonmentModel()

# Initialize model
model_path = find_model_file()
model = get_model(model_path)

# Model upload section
if model.model is None:
    st.warning("⚠️ Model not found in expected locations. Please upload your model file.")
    uploaded_model = st.file_uploader("Upload model file (PKL format)", type=["pkl"])
    
    if uploaded_model:
        # Save the uploaded model temporarily
        temp_model_path = "uploaded_model.pkl"
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        
        # Load the uploaded model
        model = CartAbandonmentModel(temp_model_path)
        if model.model:
            st.success("✅ Model uploaded and loaded successfully!")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Analyze Abandonment", "About"])

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 42px !important;
        font-weight: bold;
        color: #FF4B4B;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    .abandonment-likely {
        background-color: #FFE6E6;
        border-left: 5px solid #FF4B4B;
    }
    .purchase-likely {
        background-color: #E6FFE6;
        border-left: 5px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Home page
if page == "Home":
    st.markdown("<h1 class='main-title'>Cart Abandonment in E-commerce</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predict and prevent lost sales with machine learning</p>", unsafe_allow_html=True)
    
    st.write("""
    Welcome to the Cart Abandonment Analysis Tool! This application uses machine learning to predict 
    when customers are likely to abandon their shopping carts before completing a purchase.
    """)
    
    # Key stats
    st.subheader("📊 Cart Abandonment Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Average Abandonment Rate", value="69.8%", help="Industry average cart abandonment rate")
    with col2:
        st.metric(label="Lost Revenue", value="$18B/year", help="Estimated annual revenue lost to cart abandonment")
    with col3:
        st.metric(label="Recovery Potential", value="63%", help="Percentage of abandoned carts that are potentially recoverable")
    
    # Information section
    st.subheader("🔍 Why Customers Abandon Carts")
    
    causes = pd.DataFrame({
        'Cause': ['Extra Costs', 'Account Creation Required', 'Complex Checkout', 'Couldn\'t Calculate Costs', 'Delivery Too Slow'],
        'Percentage': [49, 24, 18, 17, 16]
    })
    
    chart = alt.Chart(causes).mark_bar().encode(
        x=alt.X('Percentage:Q', title='Percentage of Shoppers'),
        y=alt.Y('Cause:N', sort='-x', title='Abandonment Reason'),
        color=alt.Color('Percentage:Q', scale=alt.Scale(scheme='reds'), legend=None)
    ).properties(
        title='Top Reasons for Cart Abandonment',
        height=250
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # How to use
    st.subheader("How to Use This Tool")
    st.write("""
    1. Navigate to the "Analyze Abandonment" page using the sidebar
    2. Enter visitor browsing behavior and session information
    3. Click "Check Abandonment Risk" to see if the visitor is likely to abandon their cart
    4. Review the recommended actions to reduce abandonment risk
    """)
    
    # Project information
    st.info("""
    **About this Project**
    
    This tool is part of the "Cart Abandonment in E-commerce" final year project by Melly567.
    The model is trained on online shopping behavior data to identify patterns that lead to cart abandonment.
    """)

# Prediction page
elif page == "Analyze Abandonment":
    st.markdown("<h1>Cart Abandonment Analysis</h1>", unsafe_allow_html=True)
    st.write("Enter customer behavior data to predict abandonment risk.")
    
    if model.model is None:
        st.error("⚠️ Model not loaded. Please upload a model file first.")
    else:
        # Create two columns for the form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Page Interactions")
            administrative = st.number_input("Administrative Pages", min_value=0.0, value=0.0, 
                                           help="Number of administrative pages visited")
            administrative_duration = st.number_input("Administrative Duration (seconds)", min_value=0.0, value=0.0, 
                                                   help="Time spent on administrative pages")
            
            informational = st.number_input("Informational Pages", min_value=0.0, value=0.0, 
                                          help="Number of informational pages visited")
            informational_duration = st.number_input("Informational Duration (seconds)", min_value=0.0, value=0.0, 
                                                  help="Time spent on informational pages")
            
            product_related = st.number_input("Product Related Pages", min_value=0.0, value=0.0, 
                                            help="Number of product pages visited")
            product_related_duration = st.number_input("Product Related Duration (seconds)", min_value=0.0, value=0.0, 
                                                    help="Time spent on product related pages")
        
        with col2:
            st.subheader("Session Information")
            bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=1.0, value=0.2, 
                                  help="Bounce rate for the session")
            exit_rate = st.slider("Exit Rate", min_value=0.0, max_value=1.0, value=0.2, 
                                help="Exit rate for the session")
            page_value = st.number_input("Page Value", min_value=0.0, value=0.0, 
                                       help="Page value metric")
            
            special_day = st.slider("Special Day", min_value=0.0, max_value=1.0, value=0.0, 
                                  help="Closeness to special days (e.g. Mother's day, Valentine's)")
            month = st.selectbox("Month", ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 
                               help="Month of the visit")
            weekend = st.selectbox("Weekend", [True, False], 
                                 help="Whether the visit was on weekend")
        
        st.subheader("User and Technical Information")
        col3, col4 = st.columns(2)
        
        with col3:
            operating_systems = st.number_input("Operating System", min_value=1, max_value=10, value=2, 
                                              help="Operating system identifier")
            browser = st.number_input("Browser", min_value=1, max_value=20, value=2, 
                                    help="Browser identifier")
            
        with col4:
            region = st.number_input("Region", min_value=1, max_value=10, value=1, 
                                   help="Region identifier")
            traffic_type = st.number_input("Traffic Type", min_value=1, max_value=20, value=2, 
                                         help="Traffic type identifier")
            visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"], 
                                      help="Type of visitor")
        
        # Prediction button - with renamed label for cart abandonment focus
        if st.button("Check Abandonment Risk", type="primary"):
            # Prepare the features
            features = {
                'Administrative': administrative,
                'Administrative_Duration': administrative_duration,
                'Informational': informational,
                'Informational_Duration': informational_duration,
                'ProductRelated': product_related,
                'ProductRelated_Duration': product_related_duration,
                'BounceRates': bounce_rate,
                'ExitRates': exit_rate,
                'PageValues': page_value,
                'SpecialDay': special_day,
                'Month': month,
                'OperatingSystems': operating_systems,
                'Browser': browser,
                'Region': region,
                'TrafficType': traffic_type,
                'VisitorType': visitor_type,
                'Weekend': weekend,
            }
            
            # Make prediction
            try:
                with st.spinner("Analyzing customer behavior..."):
                    will_purchase = model.predict(features)
                    
                    # Invert prediction for cart abandonment (True = will purchase, False = will abandon)
                    will_abandon = not will_purchase
                    
                    # Show result
                    st.subheader("Prediction Result:")
                    
                    if will_abandon:
                        # High risk of abandonment
                        st.markdown("<div class='prediction-box abandonment-likely'>", unsafe_allow_html=True)
                        st.warning("⚠️ **HIGH RISK OF CART ABANDONMENT DETECTED**")
                        st.markdown("""
                        This visitor shows patterns consistent with customers who abandon their shopping carts.
                        
                        **Recommended interventions:**
                        
                        1. **Offer a discount code** to incentivize completion
                        2. **Show shipping cost estimates** early in the checkout process
                        3. **Implement an exit-intent popup** with a special offer
                        4. **Simplify the checkout process** - reduce form fields and steps
                        5. **Enable guest checkout** if not already available
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Low risk of abandonment
                        st.markdown("<div class='prediction-box purchase-likely'>", unsafe_allow_html=True)
                        st.success("✅ **LOW RISK OF CART ABANDONMENT**")
                        st.markdown("""
                        This visitor shows patterns consistent with customers who complete their purchases.
                        
                        **Recommended actions:**
                        
                        1. **Suggest complementary products** for cross-selling
                        2. **Highlight free shipping thresholds** if applicable
                        3. **Show estimated delivery date** prominently
                        4. **Offer warranty or protection plans** for high-value items
                        5. **Implement express checkout** for returning customers
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Feature importance visualization (sample)
                    st.subheader("Key Factors in This Prediction:")
                    
                    # This is a sample visualization - in a real implementation, you would
                    # extract actual feature importance from your model
                    feature_imp = pd.DataFrame({
                        'Feature': ['Exit Rate', 'Page Value', 'Product Duration', 'Visitor Type', 'Weekend'],
                        'Importance': [0.32, 0.28, 0.22, 0.10, 0.08]
                    })
                    
                    chart = alt.Chart(feature_imp).mark_bar().encode(
                        x=alt.X('Importance:Q'),
                        y=alt.Y('Feature:N', sort='-x'),
                        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='blues'))
                    ).properties(
                        title='Feature Importance',
                        height=200
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.text(f"Error details: {e}")

# About page
elif page == "About":
    st.title("About Cart Abandonment in E-commerce")
    
    st.write("""
    ## Project Overview
    
    Cart abandonment is one of the biggest challenges facing e-commerce businesses today, with an average 
    abandonment rate of nearly 70%. This means that for every 10 customers who add products to their cart, 
    7 will leave without completing their purchase.
    
    This tool uses machine learning to predict cart abandonment based on user behavior patterns and 
    provides actionable recommendations to reduce abandonment rates.
    
    ## The Model
    
    The prediction model is trained on online shopping behavior data, examining factors such as:
    
    - **Browsing patterns**: Number of pages visited and time spent
    - **User engagement metrics**: Bounce rate, exit rate
    - **Session value**: Potential purchase amount
    - **Temporal factors**: Month, day of week
    - **User context**: New vs. returning visitors
    
    ## Technical Implementation
    
    This application is built with:
    - **Streamlit**: For the interactive web interface
    - **Scikit-learn**: For the machine learning model
    - **Pandas & NumPy**: For data processing
    - **Altair**: For data visualization
    
    ## Project by Melissa Eberechi Onwuka
    
    This is a Final Year Project developed by Melissa Eberechi Onwuka.
    """)
    
    # References section
    st.subheader("References")
    st.markdown("""
    - Baymard Institute. (2023). *41 Cart Abandonment Rate Statistics*
    - Statista. (2023). *Online shopping cart abandonment rate worldwide*
    - SaleCycle. (2023). *The Remarketing Report – Q2 2023*
    """)

# Add footer with current date
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.markdown("---")
st.markdown(f"Cart Abandonment in E-commerce | © Melissa Onwuka | {current_time}")
