"""
Medical and Patient-Friendly Visualizations for CardioGAM-Fusion++
Creates graphs that are easily understood by doctors and normal clients
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta

def create_risk_gauge(risk_score, title="Cardiovascular Risk Assessment"):
    """
    Create an enhanced, interactive risk gauge for patients with improved styling
    """
    # Determine risk category and color with more detailed categorization
    if risk_score < 0.075:
        category = "VERY LOW RISK"
        color = "#28a745"  # Green
        description = "Excellent cardiovascular health. Continue healthy lifestyle habits."
        risk_level = "Optimal"
    elif risk_score < 0.2:
        category = "LOW RISK"
        color = "#20c997"  # Teal
        description = "Good cardiovascular health. Maintain current healthy practices."
        risk_level = "Low"
    elif risk_score < 0.3:
        category = "BORDERLINE RISK"
        color = "#ffc107"  # Yellow
        description = "Borderline risk. Consider lifestyle modifications and regular monitoring."
        risk_level = "Borderline"
    elif risk_score < 0.5:
        category = "MODERATE RISK"
        color = "#fd7e14"  # Orange
        description = "Moderate risk detected. Lifestyle changes and medical consultation recommended."
        risk_level = "Moderate"
    elif risk_score < 0.7:
        category = "HIGH RISK"
        color = "#dc3545"  # Red
        description = "High risk detected. Immediate medical attention and aggressive risk management needed."
        risk_level = "High"
    else:
        category = "VERY HIGH RISK"
        color = "#6f42c1"  # Purple
        description = "Critical risk level. Urgent cardiology consultation required."
        risk_level = "Critical"

    fig = go.Figure()

    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score * 100,
        domain={'x': [0, 1], 'y': [0.3, 1]},
        title={'text': f"<b>{title}</b><br><span style='font-size:16px;color:{color};font-weight:bold;'>{category}</span><br><span style='font-size:12px;color:gray;'>10-Year ASCVD Risk</span>"},
        delta={'reference': 7.5, 'increasing': {'color': color}, 'decreasing': {'color': '#28a745'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue", 'tickfont': {'size': 10}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 3,
            'bordercolor': color,
            'steps': [
                {'range': [0, 7.5], 'color': 'rgba(40, 167, 69, 0.3)', 'name': 'Low'},
                {'range': [7.5, 20], 'color': 'rgba(255, 193, 7, 0.3)', 'name': 'Moderate'},
                {'range': [20, 100], 'color': 'rgba(220, 53, 69, 0.3)', 'name': 'High'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.8,
                'value': risk_score * 100
            }
        },
        number={'font': {'size': 36, 'color': color}, 'suffix': '%'}
    ))

    # Add risk level indicator
    fig.add_annotation(
        text=f"<b>Risk Level: {risk_level}</b>",
        xref="paper", yref="paper",
        x=0.5, y=0.2,
        showarrow=False,
        font=dict(size=14, color=color, family="Arial, sans-serif"),
        align="center",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=color,
        borderwidth=2,
        borderpad=8
    )

    # Add detailed description
    fig.add_annotation(
        text=f"<b>Assessment:</b><br>{description}",
        xref="paper", yref="paper",
        x=0.5, y=0.05,
        showarrow=False,
        font=dict(size=11, color="black", family="Arial, sans-serif"),
        align="center",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6
    )

    # Add clinical reference
    fig.add_annotation(
        text="<i>Based on ACC/AHA 2019 Guidelines</i>",
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        showarrow=False,
        font=dict(size=8, color="gray"),
        align="right"
    )

    fig.update_layout(
        font={'family': "Arial, sans-serif", 'size': 14},
        height=500,
        margin=dict(l=20, r=20, t=80, b=100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig

def create_medical_risk_distribution(assessments_data, title="Clinical Risk Distribution"):
    """
    Create a medical-grade risk distribution chart for doctors
    """
    if not assessments_data:
        return px.histogram(title="No data available")

    df = pd.DataFrame(assessments_data)

    # Create subplots for comprehensive view
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Risk Score Distribution', 'Risk Categories', 'Age vs Risk', 'BP vs Risk'),
        specs=[[{"secondary_y": False}, {"type": "domain"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Risk score histogram
    fig.add_trace(
        go.Histogram(x=df['risk_score'], nbinsx=20, name="Risk Scores",
                    marker_color='#1f77b4', opacity=0.7),
        row=1, col=1
    )

    # Risk category pie chart
    risk_counts = df['risk_category'].value_counts()
    colors = {'Low Risk': '#28a745', 'Moderate Risk': '#ffc107', 'High Risk': '#dc3545'}
    fig.add_trace(
        go.Pie(labels=risk_counts.index, values=risk_counts.values,
               marker_colors=[colors.get(cat, '#666666') for cat in risk_counts.index],
               name="Categories"),
        row=1, col=2
    )

    # Age vs Risk scatter
    fig.add_trace(
        go.Scatter(x=df['age'], y=df['risk_score'], mode='markers',
                  marker=dict(color=df['risk_score'], colorscale='RdYlGn_r', showscale=True,
                            colorbar=dict(title="Risk Score", x=0.45, y=0.25, len=0.5)),
                  name="Age vs Risk"),
        row=2, col=1
    )

    # BP vs Risk scatter
    fig.add_trace(
        go.Scatter(x=df['bp'], y=df['risk_score'], mode='markers',
                  marker=dict(color='#ff7f0e', size=8),
                  name="BP vs Risk"),
        row=2, col=2
    )

    fig.update_layout(
        height=800,
        title_text=f"<b>{title}</b>",
        showlegend=False
    )

    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_xaxes(title_text="Age (years)", row=2, col=1)
    fig.update_xaxes(title_text="Blood Pressure (mmHg)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Risk Score", row=2, col=1)
    fig.update_yaxes(title_text="Risk Score", row=2, col=2)

    return fig

def create_patient_progress_chart(patient_history, title="Your Risk Progress"):
    """
    Create a patient-friendly progress chart showing risk changes over time
    """
    if not patient_history:
        return px.line(title="No history available")

    df = pd.DataFrame(patient_history)

    # Sort by date
    df = df.sort_values('date')

    fig = go.Figure()

    # Add risk score line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['risk_score'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8, color='#1f77b4')
    ))

    # Add goal line (target risk < 0.3)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=[0.3] * len(df),
        mode='lines',
        name='Target Risk (< 0.3)',
        line=dict(color='#28a745', width=2, dash='dash')
    ))

    # Color background based on risk zones
    fig.add_hrect(y0=0.7, y1=1.0, line_width=0, fillcolor="red", opacity=0.1, annotation_text="High Risk")
    fig.add_hrect(y0=0.3, y1=0.7, line_width=0, fillcolor="yellow", opacity=0.1, annotation_text="Moderate Risk")
    fig.add_hrect(y0=0, y1=0.3, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Low Risk")

    fig.update_layout(
        title=f"<b>{title}</b>",
        xaxis_title="Date",
        yaxis_title="Risk Score",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

def create_clinical_decision_support(risk_score, age, bp, cholesterol, hr):
    """
    Create a clinical decision support visualization for doctors
    """
    # Calculate risk factors
    factors = {
        'Age': {'value': age, 'normal': '< 50', 'elevated': '50-70', 'high': '> 70'},
        'Blood Pressure': {'value': bp, 'normal': '< 120', 'elevated': '120-140', 'high': '> 140'},
        'Cholesterol': {'value': cholesterol, 'normal': '< 200', 'elevated': '200-240', 'high': '> 240'},
        'Heart Rate': {'value': hr, 'normal': '60-100', 'elevated': '< 60 or > 100', 'high': '> 100'}
    }

    # Create radar chart for risk factors
    categories = list(factors.keys())

    fig = go.Figure()

    # Patient values (normalized)
    patient_values = []
    for factor in factors.values():
        val = factor['value']
        if factor == factors['Age']:
            patient_values.append(min(val / 80, 1))  # Normalize age
        elif factor == factors['Blood Pressure']:
            patient_values.append(min(val / 180, 1))  # Normalize BP
        elif factor == factors['Cholesterol']:
            patient_values.append(min(val / 300, 1))  # Normalize cholesterol
        else:  # Heart Rate
            patient_values.append(min(abs(val - 80) / 40 + 0.5, 1))  # Normalize HR

    # Normal ranges
    normal_values = [0.3, 0.3, 0.3, 0.5]  # Approximate normal ranges

    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        name='Patient Values',
        line_color='#1f77b4'
    ))

    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        name='Normal Range',
        line_color='#28a745',
        opacity=0.5
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="<b>Clinical Risk Factors Assessment</b>",
        height=500
    )

    return fig

def create_population_comparison(risk_score, age, gender='unknown', title="Population Risk Comparison"):
    """
    Create a visualization comparing patient risk to population percentiles
    """
    # Simulate population data (in real app, this would come from epidemiological data)
    population_data = {
        'age_groups': ['30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
        'median_risk': [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
        'percentile_75': [0.25, 0.35, 0.45, 0.55, 0.65, 0.75],
        'percentile_90': [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    }

    # Determine age group
    if age < 40:
        age_group = '30-39'
        idx = 0
    elif age < 50:
        age_group = '40-49'
        idx = 1
    elif age < 60:
        age_group = '50-59'
        idx = 2
    elif age < 70:
        age_group = '60-69'
        idx = 3
    elif age < 80:
        age_group = '70-79'
        idx = 4
    else:
        age_group = '80+'
        idx = 5

    fig = go.Figure()

    # Add population distribution
    fig.add_trace(go.Bar(
        x=population_data['age_groups'],
        y=population_data['median_risk'],
        name='Population Median',
        marker_color='#1f77b4',
        opacity=0.7
    ))

    # Add patient risk line
    fig.add_trace(go.Scatter(
        x=[age_group],
        y=[risk_score],
        mode='markers',
        name='Your Risk',
        marker=dict(color='#dc3545', size=12, symbol='diamond')
    ))

    # Add percentile lines
    fig.add_trace(go.Scatter(
        x=population_data['age_groups'],
        y=population_data['percentile_75'],
        mode='lines',
        name='75th Percentile',
        line=dict(color='#ffc107', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=population_data['age_groups'],
        y=population_data['percentile_90'],
        mode='lines',
        name='90th Percentile',
        line=dict(color='#dc3545', dash='dot')
    ))

    fig.update_layout(
        title=f"<b>{title}</b><br><sub>Age Group: {age_group}</sub>",
        xaxis_title="Age Groups",
        yaxis_title="Risk Score",
        height=400,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig

def create_lifestyle_recommendations(risk_score, age, bp, cholesterol, hr):
    """
    Create personalized lifestyle recommendations visualization
    """
    recommendations = []

    # Risk-based recommendations
    if risk_score < 0.3:
        recommendations.append(("✅ Excellent! Continue healthy habits", "success"))
    elif risk_score < 0.7:
        recommendations.append(("⚠️ Moderate risk - Lifestyle changes recommended", "warning"))
    else:
        recommendations.append(("🚨 High risk - Immediate medical attention needed", "danger"))

    # Specific recommendations based on vitals
    if bp > 140:
        recommendations.append(("💊 High blood pressure - Consult doctor about medication", "danger"))
    elif bp > 120:
        recommendations.append(("🧘 Reduce salt intake and exercise regularly", "warning"))

    if cholesterol > 240:
        recommendations.append(("🥗 High cholesterol - Dietary changes needed", "danger"))
    elif cholesterol > 200:
        recommendations.append(("🥑 Monitor cholesterol levels", "warning"))

    if hr > 100 or hr < 60:
        recommendations.append(("🏃‍♂️ Irregular heart rate - Cardiovascular exercise recommended", "warning"))

    if age > 50:
        recommendations.append(("📅 Annual cardiovascular check-ups recommended", "info"))

    # Create visualization
    fig = go.Figure()

    # Add recommendations as annotations
    y_pos = 0.9
    for rec, color in recommendations:
        color_map = {'success': '#28a745', 'warning': '#ffc107', 'danger': '#dc3545', 'info': '#17a2b8'}

        fig.add_annotation(
            text=rec,
            xref="paper", yref="paper",
            x=0.02, y=y_pos,
            showarrow=False,
            font=dict(size=12, color=color_map.get(color, '#000000')),
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color_map.get(color, '#000000'),
            borderwidth=1,
            borderpad=4
        )
        y_pos -= 0.15

    fig.update_layout(
        title="<b>Personalized Health Recommendations</b>",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

def create_medical_summary_card(risk_score, age, bp, cholesterol, hr, confidence):
    """
    Create a medical summary card with key metrics
    """
    # Determine risk category
    if risk_score < 0.3:
        category = "Low Risk"
        color = "#28a745"
        icon = "🟢"
    elif risk_score < 0.7:
        category = "Moderate Risk"
        color = "#ffc107"
        icon = "🟡"
    else:
        category = "High Risk"
        color = "#dc3545"
        icon = "🔴"

    fig = go.Figure()

    # Add background
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        fillcolor=color,
        opacity=0.1,
        line=dict(color=color, width=2)
    )

    # Add title
    fig.add_annotation(
        text=f"<b>{icon} {category}</b>",
        x=0.5, y=0.9,
        showarrow=False,
        font=dict(size=20, color=color),
        xref="paper", yref="paper"
    )

    # Add risk score
    fig.add_annotation(
        text=f"<b>Risk Score: {risk_score:.3f}</b>",
        x=0.5, y=0.7,
        showarrow=False,
        font=dict(size=16),
        xref="paper", yref="paper"
    )

    # Add confidence
    fig.add_annotation(
        text=f"Confidence: {confidence}",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
        xref="paper", yref="paper"
    )

    # Add key vitals
    vitals_text = f"""
    <b>Key Vitals:</b><br>
    Age: {age} years<br>
    BP: {bp} mmHg<br>
    Cholesterol: {cholesterol} mg/dL<br>
    Heart Rate: {hr} bpm
    """

    fig.add_annotation(
        text=vitals_text,
        x=0.02, y=0.3,
        showarrow=False,
        font=dict(size=12),
        xref="paper", yref="paper",
        align="left"
    )

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    )

    return fig
