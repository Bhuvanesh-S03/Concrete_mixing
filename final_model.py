import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

# Load the dataset
data_path = './Data_for_Opt.csv'
data = pd.read_csv(data_path)

# Handle missing values
if data.isnull().any().any():
    st.warning("Looks like some data is missing. We’ve cleaned it up for better predictions.")
    data = data.dropna()

# Define features and targets
features = data.drop(columns=['Age', 'Strength(MPa)', 'CO2(kg)', 'Cost(USD)'])
targets = data[['Strength(MPa)', 'CO2(kg)', 'Cost(USD)']]
data_description = data.describe()
min_max_values = data_description.loc[['min', 'max']].transpose()

# Train models for each target
models = {target: LinearRegression().fit(features, targets[target]) for target in targets.columns}

# Prediction function
def predict(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    return {target: models[target].predict(input_array)[0] for target in targets.columns}

# Streamlit App
def main():
    st.title("🔎 Concrete Mix Optimizer")
    st.write("Enter your mix proportions and let AI predict the strength, cost, and environmental impact.")

    with st.sidebar:
        st.header("🔧 Adjust Mix Proportions")
        inputs = {
            feature: st.slider(
                feature,
                float(min_max_values.loc[feature, 'min']),
                float(min_max_values.loc[feature, 'max']),
                value=(min_max_values.loc[feature, 'min'] + min_max_values.loc[feature, 'max']) / 2
            ) for feature in features.columns
        }

    if st.button("🚀 Get Predictions"):
        predictions = predict(list(inputs.values()))
        
        st.subheader("📊 Predicted Results")
        st.write(f"**Strength:** {predictions['Strength(MPa)']:.2f} MPa")
        st.write(f"**CO₂ Emissions:** {predictions['CO2(kg)']:.2f} kg")
        st.write(f"**Estimated Cost:** ${predictions['Cost(USD)']:.2f}")

        # Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=list(predictions.keys()), y=list(predictions.values()), ax=ax)
        ax.set_ylabel("Values")
        ax.set_title("Breakdown of Predictions")
        st.pyplot(fig)

        # 3D Scatter Plot
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=[predictions['Strength(MPa)']],
            y=[predictions['CO2(kg)']],
            z=[predictions['Cost(USD)']],
            mode='markers',
            marker=dict(size=10, color='red', opacity=0.8),
            name='Predicted Values'
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Strength (MPa)',
                yaxis_title='CO2 (kg)',
                zaxis_title='Cost (USD)'
            ),
            title="🌎 3D Visualization of Predictions"
        )
        st.plotly_chart(fig_3d)

if __name__ == "__main__":
    main()
s