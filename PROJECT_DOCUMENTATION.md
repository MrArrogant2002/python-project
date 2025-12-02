# NYC Taxi Fare Prediction Project 🚕

## Project Overview

A comprehensive machine learning project for predicting NYC taxi fares using advanced regression techniques, featuring an interactive web application and detailed data analysis.

---

## Table of Contents

1. [Project Summary](#project-summary)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Development Journey](#development-journey)
5. [Model Development](#model-development)
6. [Web Application](#web-application)
7. [Data Analysis](#data-analysis)
8. [Key Features](#key-features)
9. [Performance Metrics](#performance-metrics)
10. [How to Use](#how-to-use)

---

## Project Summary

This project predicts taxi fare amounts in New York City based on various factors including:
- Pickup and dropoff locations (latitude/longitude)
- Pickup date and time
- Passenger count
- Trip distance
- Temporal features (rush hours, weekends, business hours)
- Proximity to major landmarks (JFK, LaGuardia, Newark airports)

**Goal:** Build an accurate fare prediction system with an intuitive user interface for real-time predictions.

---

## Technology Stack

### Core Technologies
- **Python 3.13.6** - Primary programming language
- **Virtual Environment** - Isolated development environment

### Machine Learning & Data Science
- **XGBoost** - Gradient boosting algorithm for regression
- **scikit-learn** - Machine learning pipeline and preprocessing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations

### Web Development
- **Streamlit 1.51.0** - Interactive web application framework
- **Session State** - UI state management

### Data Visualization
- **matplotlib** - Static plots and charts
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **folium** (planned) - Interactive map integration

### Development Tools
- **Jupyter Notebook** - Interactive development and analysis
- **Git** - Version control
- **GitHub** - Code repository hosting

---

## Project Structure

```
z-python-project/
├── app.py                              # Streamlit web application (815 lines)
├── enhanced-taxi-fare-model.ipynb      # Model training notebook
├── data-visualization-analysis.ipynb   # Data analysis notebook
├── train.csv                           # Training dataset
├── test.csv                            # Test dataset
├── pipeline_submission.csv             # Model predictions
├── taxi_fare_enhanced_model.pkl        # Trained model (8.7 MB)
├── requirements.txt                    # Python dependencies
├── venv/                               # Virtual environment
├── BUG_FIXES.md                        # Bug fix documentation
├── IMPLEMENTATION_COMPLETE.md          # Implementation notes
└── PROJECT_DOCUMENTATION.md            # This file
```

---

## Development Journey

### Phase 1: Initial Setup
1. **Environment Setup**
   - Created Python 3.13.6 virtual environment
   - Installed core dependencies (pandas, numpy, scikit-learn, xgboost)
   
2. **Data Acquisition**
   - Loaded NYC taxi fare dataset (train.csv, test.csv)
   - Dataset contains 8 features and 500K+ records

### Phase 2: Model Development
1. **Baseline Model**
   - Simple regression with basic features
   - Initial accuracy: ~75%

2. **Feature Engineering**
   - Extracted temporal features (hour, day, month, weekday)
   - Calculated haversine distance between pickup/dropoff
   - Created binary indicators:
     - Morning rush (7-9 AM)
     - Evening rush (5-7 PM)
     - Weekend flag
     - Late night (11 PM - 5 AM)
     - Business hours (9 AM - 5 PM)
   - Computed distances to major landmarks (JFK, LaGuardia, EWR, Manhattan center)

3. **Enhanced Model**
   - Implemented XGBoost with optimized hyperparameters
   - Added custom transformer pipeline
   - Total features: 26
   - **Final Performance:**
     - Training RMSE: $3.05
     - Validation RMSE: $3.57
     - Training R²: 90.14%
     - Validation R²: 86.41%

### Phase 3: Web Application Development
1. **Initial App Creation**
   - Built Streamlit interface with input forms
   - Implemented prediction functionality
   - Added preset locations for quick testing

2. **UI/UX Improvements**
   - Fixed text visibility issues (dark theme compatibility)
   - Added custom CSS for better contrast
   - Implemented map preview with pickup/dropoff visualization
   - Added route distance display

3. **Bug Fixes & Enhancements**
   - Fixed time input reset issue using session state
   - Changed metric colors to black for visibility
   - Removed redundant model version display
   - Enhanced error handling and validation

4. **Advanced Features**
   - Comprehensive input validation (NYC bounds checking)
   - Same location prevention
   - Model performance metrics display
   - Interactive map preview
   - Preset location shortcuts

### Phase 4: Data Analysis & Visualization
1. **Comprehensive Analysis Notebook**
   - Created 10-section analysis workflow
   - Statistical summaries and distributions
   - Temporal pattern analysis
   - Geographic heatmaps
   - Correlation studies
   - Model performance evaluation

2. **Optimization**
   - Implemented efficient data loading (skiprows method)
   - Sample-based visualization for performance
   - Reduced loading time from minutes to seconds

---

## Model Development

### Custom Transformer Classes

**1. EnhancedDatetimeFeatureExtractor**
- Extracts temporal features from pickup datetime
- Creates 5 binary indicators for time-based patterns
- Features: hour, day_of_week, month, morning_rush, evening_rush, weekend, late_night, business_hours

**2. DistanceCalculator**
- Computes haversine distance between pickup/dropoff
- Accounts for Earth's curvature
- Output in kilometers

**3. LandmarkDistanceCalculator**
- Calculates distance from pickup to major landmarks
- Landmarks: JFK Airport, LaGuardia, Newark, Manhattan Center
- Helps identify airport routes and downtown trips

**4. OutlierRemover**
- Filters invalid/extreme values
- Fare range: $0-$250
- NYC geographic bounds validation
- Passenger count: 1-6

**5. EnhancedFeatureSelector**
- Selects relevant features for modeling
- Excludes non-predictive columns (key, datetime)

### Model Pipeline
```python
Pipeline([
    ('datetime_features', EnhancedDatetimeFeatureExtractor()),
    ('distance', DistanceCalculator()),
    ('landmarks', LandmarkDistanceCalculator()),
    ('outliers', OutlierRemover()),
    ('selector', EnhancedFeatureSelector()),
    ('model', XGBRegressor(parameters...))
])
```

### Hyperparameters
- n_estimators: 200
- max_depth: 8
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- reg_alpha: 0.1 (L1 regularization)
- reg_lambda: 1.0 (L2 regularization)

---

## Web Application

### Features

**Input Methods:**
1. **Manual Input**
   - Pickup/dropoff coordinates
   - Date and time selection
   - Passenger count

2. **Preset Locations** (16 popular routes)
   - Times Square ↔ JFK Airport
   - Manhattan ↔ LaGuardia Airport
   - Central Park ↔ Brooklyn Bridge
   - Empire State Building routes
   - And more...

**Visualization:**
- Interactive map preview showing:
  - Pickup location (green marker)
  - Dropoff location (red marker)
  - Direct route line (blue)
  - Distance calculation
  - NYC landmarks overlay

**Prediction Display:**
- Large, prominent fare prediction
- Model performance metrics
- Trip details summary
- Distance information

**Validation:**
- NYC bounds checking (lat: 40.60-40.90, lon: -74.05 to -73.75)
- Same location prevention
- Valid passenger count (1-6)
- Date range validation

### Technical Implementation

**Session State Management:**
- Preserves time input across reruns
- Maintains UI selections
- Prevents unwanted resets

**CSS Customization:**
- Dark theme compatibility
- High contrast text (black metrics)
- Custom styled info boxes
- Responsive layout

**Error Handling:**
- Graceful model loading fallback
- User-friendly error messages
- Input validation feedback

---

## Data Analysis

### Analysis Notebook Sections

**1. Data Loading & Exploration**
- Efficient loading with 10% sampling (500K rows)
- Missing value analysis
- Data type verification

**2. Statistical Summary**
- Descriptive statistics
- Fare distribution analysis
- IQR and outlier detection

**3. Fare Distribution Analysis**
- Histogram visualization
- Box plot for outliers
- Density plot (KDE)
- Q-Q plot for normality test
- **Key Finding:** Right-skewed distribution, median $8.50

**4. Temporal Patterns**
- Hourly fare averages
- Day of week analysis
- Monthly trends
- Trip count by hour
- **Key Finding:** Rush hour peaks at 5-6 AM and 5-7 PM

**5. Geographic Analysis**
- Pickup location heatmap
- Dropoff location heatmap
- Density visualization
- **Key Finding:** Heavy Manhattan concentration

**6. Distance vs Fare Analysis**
- Scatter plot with trend line
- Distance distribution histogram
- Correlation calculation
- Price per km analysis
- **Key Finding:** 0.88 correlation (very strong)

**7. Passenger Count Impact**
- Average fare by passenger count
- Distribution pie chart
- **Key Finding:** 69.1% single passenger trips

**8. Model Performance Evaluation**
- Loads trained model
- Displays all metrics
- Shows model enhancements
- **Key Finding:** 86.41% R² accuracy

**9. Key Insights Summary**
- Business recommendations
- Pattern identification
- Actionable insights

### Data Insights

**Fare Patterns:**
- Most fares: $5-$20 range
- Mean: $11.32
- Median: $8.50
- Right-skewed distribution

**Temporal Patterns:**
- Morning rush: 7-9 AM (higher fares)
- Evening rush: 5-7 PM (peak demand)
- Late night: 11 PM - 5 AM (surcharges)
- Weekend patterns differ from weekdays

**Geographic Patterns:**
- Manhattan: Highest density
- Airport routes: Common (JFK, LaGuardia, Newark)
- Most trips: Under 10 km

**Distance Impact:**
- Strong correlation: 0.88
- Average trip: 3.27 km
- Median trip: 2.17 km
- Price per km varies by time/location

---

## Key Features

### 1. Accurate Predictions
- 86.41% R² accuracy
- $3.57 RMSE on validation set
- Considers 26+ features

### 2. User-Friendly Interface
- Clean, intuitive design
- Preset locations for convenience
- Real-time predictions
- Visual map preview

### 3. Comprehensive Analysis
- 10-section data analysis
- Multiple visualization types
- Statistical insights
- Business recommendations

### 4. Production-Ready Code
- Error handling
- Input validation
- Session state management
- Optimized performance

### 5. Model Enhancements
- Rush hour detection
- Weekend/weekday differentiation
- Late night surcharges
- Business hours tracking
- Landmark proximity
- Distance calculations

---

## Performance Metrics

### Model Performance
| Metric | Training | Validation |
|--------|----------|------------|
| RMSE | $3.05 | $3.57 |
| MAE | $1.45 | $1.53 |
| R² Score | 0.9014 (90.14%) | 0.8641 (86.41%) |

### Data Loading
- Original: Minutes for full dataset
- Optimized: ~5.6 seconds for 500K rows (10% sample)
- Method: Skiprows with sampling

### Application Performance
- Model loading: < 1 second
- Prediction time: < 0.1 seconds
- Map rendering: < 0.5 seconds

---

## How to Use

### Installation

1. **Clone Repository**
```bash
git clone <repository-url>
cd z-python-project
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or
venv\Scripts\activate  # Windows CMD
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

**Web App:**
```bash
streamlit run app.py
```
Access at: http://localhost:8501

**Analysis Notebook:**
```bash
jupyter notebook data-visualization-analysis.ipynb
```

**Model Training:**
```bash
jupyter notebook enhanced-taxi-fare-model.ipynb
```

### Making Predictions

**Method 1: Preset Locations**
1. Open web app
2. Click "Select from Preset Locations"
3. Choose a popular route
4. Click "Predict Fare"

**Method 2: Manual Input**
1. Enter pickup coordinates
2. Enter dropoff coordinates
3. Select date and time
4. Choose passenger count
5. Click "Predict Fare"

### Analyzing Data

1. Open `data-visualization-analysis.ipynb`
2. Run cells sequentially
3. View generated visualizations
4. Explore insights and metrics

---

## Business Recommendations

Based on data analysis:

1. **Dynamic Pricing**
   - Implement surge pricing during rush hours
   - Higher rates for airport routes
   - Weekend premium pricing

2. **Route Optimization**
   - Focus on Manhattan routes (highest demand)
   - Prioritize airport connections
   - Optimize for short trips (under 10 km)

3. **Driver Allocation**
   - Increase drivers during rush hours (7-9 AM, 5-7 PM)
   - Late night availability for premium fares
   - Strategic positioning near airports

4. **Predictive Maintenance**
   - Schedule maintenance during low-demand periods
   - Use temporal patterns for planning
   - Minimize downtime during peak hours

5. **Customer Incentives**
   - Discounts during off-peak hours
   - Loyalty programs for frequent riders
   - Bundle deals for airport trips

---

## Future Enhancements

### Planned Features
- [ ] Interactive map with click-to-select locations
- [ ] Real-time traffic data integration
- [ ] Weather impact analysis
- [ ] Multiple model comparison
- [ ] Historical fare trends
- [ ] Mobile-responsive design
- [ ] API endpoint for predictions
- [ ] Batch prediction support

### Technical Improvements
- [ ] Model retraining pipeline
- [ ] A/B testing framework
- [ ] Performance monitoring
- [ ] Automated testing suite
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)

---

## Challenges Overcome

1. **Data Loading Performance**
   - Problem: Large CSV files taking minutes to load
   - Solution: Skiprows sampling method (10x faster)

2. **Text Visibility Issues**
   - Problem: White text on light backgrounds
   - Solution: Custom CSS with !important overrides

3. **Time Input Reset**
   - Problem: Input reverting to current time
   - Solution: Session state management

4. **Model Pickle Dependencies**
   - Problem: Custom transformers not found during unpickling
   - Solution: Define all classes before loading

5. **Geographic Outliers**
   - Problem: Invalid coordinates skewing predictions
   - Solution: NYC bounds validation and filtering

---

## Contributors

- **Development Team**: Full-stack ML development
- **Data Source**: NYC Taxi & Limousine Commission

---

## License

This project is for educational and demonstration purposes.

---

## Acknowledgments

- NYC TLC for providing the dataset
- Streamlit team for the amazing framework
- XGBoost developers for the powerful ML library
- Open-source community for supporting libraries

---

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact the development team

---

**Last Updated:** December 2, 2025

**Project Status:** ✅ Complete & Production-Ready

---

## Quick Stats

- **Lines of Code:** 2000+
- **Model Accuracy:** 86.41% R²
- **Features Engineered:** 26
- **Preset Locations:** 16
- **Visualizations:** 15+
- **Development Time:** Multiple iterations
- **Files Created:** 10+

---

*Built with ❤️ using Python, Streamlit, and XGBoost*
