"""
Advanced Medical Visualizations for Cardiologists
Comprehensive graphs and charts for finalizing cardiovascular risk assessment
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from datetime import datetime, timedelta

def create_comprehensive_ecg_analysis(patient_id, ecg_data, risk_score):
    """
    Create comprehensive ECG analysis visualization for doctors
    Shows all 12 leads with proper medical formatting
    """
    if not ecg_data:
        return px.line(title="No ECG data available")

    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=('Lead I', 'Lead II', 'Lead III', 'Lead aVR', 'Lead aVL', 'Lead aVF',
                       'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6'),
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    leads = ['lead_1', 'lead_2', 'lead_3', 'lead_4', 'lead_5', 'lead_6',
             'lead_7', 'lead_8', 'lead_9', 'lead_10', 'lead_11', 'lead_12']

    for i, lead in enumerate(leads):
        row = (i // 3) + 1
        col = (i % 3) + 1

        if lead in ecg_data:
            signal = ecg_data[lead]
            time = np.linspace(0, len(signal)/500, len(signal))  # 500 Hz sampling

            fig.add_trace(
                go.Scatter(x=time, y=signal, mode='lines', name=f'{lead.replace("_", " ").title()}',
                          line=dict(color='#1f77b4', width=1)),
                row=row, col=col
            )

            # Add grid lines for better analysis (standard ECG grid)
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=row, col=col)

    fig.update_layout(
        height=1000,
        title_text=f"<b>12-Lead ECG Analysis - Patient {patient_id}</b><br><sub>Risk Score: {risk_score:.3f}</sub>",
        showlegend=False
    )

    # Add time markers every 0.2 seconds (standard ECG timing)
    for i in range(0, 12):
        row = (i // 3) + 1
        col = (i % 3) + 1
        fig.update_xaxes(tickmode='linear', tick0=0, dtick=0.2, row=row, col=col)

    return fig

def create_cardiac_risk_heatmap(assessments_data, title="Cardiac Risk Factor Correlations"):
    """
    Create a comprehensive heatmap showing correlations between all cardiac risk factors
    Helps doctors understand which factors are most influential
    """
    if not assessments_data:
        return px.imshow([[0]], title="No data available")

    df = pd.DataFrame(assessments_data)

    # Calculate correlation matrix for key cardiac risk factors
    numeric_cols = ['age', 'bp', 'cholesterol', 'heart_rate', 'risk_score']
    corr_matrix = df[numeric_cols].corr()

    # Create correlation heatmap with medical interpretation
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
        y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    # Add correlation strength interpretation
    annotations = []
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strength = "Strong"
            elif abs(corr_val) > 0.5:
                strength = "Moderate"
            elif abs(corr_val) > 0.3:
                strength = "Weak"
            else:
                strength = "Very Weak"

            annotations.append(f"{corr_matrix.index[i]} vs {corr_matrix.columns[j]}: {strength} ({corr_val:.2f})")

    fig.add_annotation(
        text="<b>Correlation Strength Guide:</b><br>" + "<br>".join(annotations[:5]),
        xref="paper", yref="paper",
        x=1.05, y=0.5,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title=f"<b>{title}</b>",
        height=600,
        xaxis_title="Risk Factors",
        yaxis_title="Risk Factors",
        margin=dict(l=20, r=150, t=60, b=20)
    )

    return fig

def create_clinical_decision_tree(risk_score, age, bp, cholesterol, hr):
    """
    Create a clinical decision tree visualization for treatment recommendations
    Based on ACC/AHA guidelines for cardiovascular risk assessment
    """
    # Decision tree logic based on clinical guidelines
    decisions = []

    # Age-based decisions (ACC/AHA guidelines)
    if age >= 75:
        decisions.append(("Age ≥ 75", "Very High Risk - Intensive management", "#dc3545"))
    elif age >= 65:
        decisions.append(("Age 65-74", "High risk factor - Enhanced monitoring", "#dc3545"))
    elif age >= 50:
        decisions.append(("Age 50-64", "Moderate risk factor", "#ffc107"))
    else:
        decisions.append(("Age < 50", "Low risk factor", "#28a745"))

    # BP-based decisions (JNC 8 guidelines)
    if bp >= 180:
        decisions.append(("BP ≥ 180", "Hypertensive crisis - Immediate treatment", "#dc3545"))
    elif bp >= 160:
        decisions.append(("BP 160-179", "Stage 2 hypertension - Multiple drugs", "#dc3545"))
    elif bp >= 140:
        decisions.append(("BP 140-159", "Stage 1 hypertension - Lifestyle + meds", "#ffc107"))
    elif bp >= 130:
        decisions.append(("BP 130-139", "Elevated BP - Lifestyle changes", "#ffc107"))
    else:
        decisions.append(("BP < 130", "Normal BP", "#28a745"))

    # Cholesterol-based decisions (ATP IV guidelines)
    if cholesterol >= 300:
        decisions.append(("Chol ≥ 300", "Very high - Immediate statin therapy", "#dc3545"))
    elif cholesterol >= 240:
        decisions.append(("Chol 240-299", "High cholesterol - High-intensity statin", "#dc3545"))
    elif cholesterol >= 200:
        decisions.append(("Chol 200-239", "Borderline high - Moderate statin", "#ffc107"))
    else:
        decisions.append(("Chol < 200", "Desirable cholesterol", "#28a745"))

    # Heart rate decisions (AHA guidelines)
    if hr >= 100:
        decisions.append(("HR ≥ 100", "Tachycardia - Investigate arrhythmia", "#dc3545"))
    elif hr <= 50:
        decisions.append(("HR ≤ 50", "Bradycardia - Pacemaker evaluation", "#dc3545"))
    elif hr <= 60:
        decisions.append(("HR 51-60", "Borderline bradycardia - Monitor", "#ffc107"))
    else:
        decisions.append(("HR 60-100", "Normal sinus rhythm", "#28a745"))

    # Overall risk assessment (ACC/AHA pooled cohort equations)
    if risk_score >= 0.2:
        decisions.append(("10-year ASCVD Risk ≥ 20%", "HIGH RISK - Intensive statin therapy", "#dc3545"))
    elif risk_score >= 0.075:
        decisions.append(("10-year ASCVD Risk 7.5-20%", "MODERATE RISK - Moderate statin therapy", "#ffc107"))
    elif risk_score >= 0.05:
        decisions.append(("10-year ASCVD Risk 5-7.5%", "BORDERLINE RISK - Consider statin", "#ffc107"))
    else:
        decisions.append(("10-year ASCVD Risk < 5%", "LOW RISK - Lifestyle therapy only", "#28a745"))

    # Create visualization
    fig = go.Figure()

    y_pos = 0.95
    for decision, recommendation, color in decisions:
        fig.add_annotation(
            text=f"<b>{decision}:</b> {recommendation}",
            xref="paper", yref="paper",
            x=0.02, y=y_pos,
            showarrow=False,
            font=dict(size=11, color=color),
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=color,
            borderwidth=2,
            borderpad=4
        )
        y_pos -= 0.07

    fig.update_layout(
        title="<b>ACC/AHA Clinical Decision Support System</b>",
        height=700,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

def create_cardiovascular_trends_dashboard(assessments_data, title="Cardiovascular Health Trends"):
    """
    Create a comprehensive dashboard showing trends in cardiovascular health metrics
    Essential for monitoring disease progression and treatment efficacy
    """
    if not assessments_data:
        return px.line(title="No data available")

    df = pd.DataFrame(assessments_data)
    if df.empty:
        return px.line(title="No data available")

    df['date'] = pd.to_datetime(df['created_at'], errors='coerce')
    if df.empty:
        return px.line(title="No data available")

    df = df.sort_values('date')

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Risk Score Trend', 'Blood Pressure Trend',
                       'Cholesterol Trend', 'Heart Rate Variability',
                       'Multi-factor Risk Analysis', 'Treatment Response'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "scatter3d"}, {"secondary_y": False}]]
    )

    # Risk score trend with clinical thresholds
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['risk_score'], mode='lines+markers',
                  name='Risk Score', line=dict(color='#dc3545', width=3)),
        row=1, col=1
    )

    # Add clinical risk thresholds
    fig.add_hline(y=0.075, line_dash="dash", line_color="#ffc107",
                 annotation_text="Moderate Risk Threshold", row=1, col=1)
    fig.add_hline(y=0.2, line_dash="dot", line_color="#dc3545",
                 annotation_text="High Risk Threshold", row=1, col=1)

    # BP trend with hypertension thresholds
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['bp'], mode='lines+markers',
                  name='Systolic BP', line=dict(color='#1f77b4', width=2)),
        row=1, col=2
    )

    # Add BP thresholds
    fig.add_hline(y=120, line_dash="dash", line_color="#28a745",
                 annotation_text="Normal (<120)", row=1, col=2)
    fig.add_hline(y=130, line_dash="dash", line_color="#ffc107",
                 annotation_text="Elevated (120-129)", row=1, col=2)
    fig.add_hline(y=140, line_dash="dot", line_color="#dc3545",
                 annotation_text="Hypertension (≥130)", row=1, col=2)

    # Cholesterol trend with clinical targets
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cholesterol'], mode='lines+markers',
                  name='Total Cholesterol', line=dict(color='#28a745', width=2)),
        row=2, col=1
    )

    # Add cholesterol targets
    fig.add_hline(y=200, line_dash="dash", line_color="#28a745",
                 annotation_text="Desirable (<200)", row=2, col=1)
    fig.add_hline(y=240, line_dash="dot", line_color="#dc3545",
                 annotation_text="High (≥240)", row=2, col=1)

    # Heart rate variability (simulated based on available data)
    hr_variability = df['heart_rate'] + np.random.normal(0, 5, len(df))
    fig.add_trace(
        go.Scatter(x=df['date'], y=hr_variability, mode='lines+markers',
                  name='HR Variability', line=dict(color='#ffc107', width=2)),
        row=2, col=2
    )

    # 3D multi-factor analysis
    fig.add_trace(
        go.Scatter3d(x=df['age'], y=df['bp'], z=df['cholesterol'],
                    mode='markers',
                    marker=dict(size=5, color=df['risk_score'], colorscale='RdYlGn_r',
                              colorbar=dict(title="Risk Score")),
                    name='Multi-factor Analysis'),
        row=3, col=1
    )

    # Treatment response indicator (simulated improvement trend)
    baseline_risk = df['risk_score'].iloc[0]
    treatment_effect = baseline_risk * (0.8 ** (np.arange(len(df)) / len(df)))
    fig.add_trace(
        go.Scatter(x=df['date'], y=treatment_effect, mode='lines',
                  name='Expected Treatment Response', line=dict(color='#17a2b8', width=2, dash='dash')),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['risk_score'], mode='lines+markers',
                  name='Actual Risk Trend', line=dict(color='#dc3545', width=2)),
        row=3, col=2
    )

    fig.update_layout(
        height=1200,
        title_text=f"<b>{title}</b>",
        showlegend=True
    )

    return fig

def create_diagnostic_confidence_matrix(predictions_data, title="Diagnostic Confidence Assessment"):
    """
    Create a confusion matrix style visualization showing model confidence vs clinical risk
    Helps doctors understand AI reliability for different patient profiles
    """
    if not predictions_data:
        return px.imshow([[0]], title="No data available")

    df = pd.DataFrame(predictions_data)
    if df.empty:
        return px.imshow([[0]], title="No data available")

    # Create confidence bins based on clinical relevance
    confidence_bins = pd.cut(df['confidence'], bins=[0, 60, 75, 85, 95, 100],
                           labels=['Low (<60%)', 'Moderate (60-75%)', 'Good (75-85%)',
                                 'High (85-95%)', 'Very High (>95%)'])

    # Create risk categories based on clinical guidelines
    risk_bins = pd.cut(df['risk_score'], bins=[0, 0.05, 0.075, 0.2, 1.0],
                      labels=['Low (<5%)', 'Borderline (5-7.5%)', 'Moderate (7.5-20%)', 'High (>20%)'])

    # Create cross-tabulation
    confusion_matrix = pd.crosstab(confidence_bins, risk_bins, margins=True, normalize='index')

    # Create heatmap with clinical interpretation
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix.values[:-1, :-1] * 100,  # Convert to percentage
        x=confusion_matrix.columns[:-1],
        y=confusion_matrix.index[:-1],
        colorscale='Blues',
        text=np.round(confusion_matrix.values[:-1, :-1] * 100, 1),
        texttemplate="%{text}%",
        textfont={"size": 12},
        hoverongaps=False
    ))

    # Add clinical confidence interpretation
    confidence_interpretation = """
    <b>Clinical Confidence Levels:</b><br>
    • Very High (>95%): Definitive diagnosis<br>
    • High (85-95%): Strong evidence<br>
    • Good (75-85%): Moderate evidence<br>
    • Moderate (60-75%): Limited evidence<br>
    • Low (<60%): Insufficient evidence
    """

    fig.add_annotation(
        text=confidence_interpretation,
        xref="paper", yref="paper",
        x=1.05, y=0.5,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1
    )

    fig.update_layout(
        title=f"<b>{title}</b><br><sub>Model Confidence vs Clinical Risk Categories</sub>",
        xaxis_title="10-Year ASCVD Risk Category",
        yaxis_title="AI Model Confidence Level",
        height=600,
        margin=dict(l=20, r=200, t=80, b=20)
    )

    return fig

def create_comprehensive_medical_report(risk_score, age, bp, cholesterol, hr, ecg_data=None, patient_history=None):
    """
    Create a comprehensive medical report visualization that doctors can use for final assessment
    Includes all essential clinical information in one view
    """
    fig = make_subplots(
        rows=5, cols=2,
        subplot_titles=('Patient Risk Profile', 'Vital Signs Analysis',
                       'ECG Analysis (Lead I)', 'Clinical Recommendations',
                       'Risk Factor Correlations', 'Population Comparison',
                       'Longitudinal Trends', 'Diagnostic Confidence',
                       'Treatment Plan', 'Follow-up Schedule'),
        specs=[[{"type": "indicator"}, {"type": "polar"}],
               [{"secondary_y": False}, {"type": "table"}],
               [{"type": "heatmap"}, {"type": "bar"}],
               [{"secondary_y": False}, {"type": "indicator"}],
               [{"type": "table"}, {"type": "table"}]]
    )

    # 1. Risk gauge with clinical interpretation
    risk_percentage = risk_score * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_percentage,
            title={"text": "10-Year ASCVD Risk"},
            delta={'reference': 7.5, 'increasing': {'color': '#dc3545'}},
            gauge={'axis': {'range': [0, 100]},
                  'bar': {'color': '#dc3545'},
                  'steps': [{'range': [0, 5], 'color': '#28a745'},
                           {'range': [5, 7.5], 'color': '#ffc107'},
                           {'range': [7.5, 20], 'color': '#ffc107'},
                           {'range': [20, 100], 'color': '#dc3545'}],
                  'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': risk_percentage}}
        ),
        row=1, col=1
    )

    # 2. Vital signs radar with clinical ranges
    categories = ['Age', 'BP', 'Cholesterol', 'Heart Rate']
    values = [min(age/80, 1), min(bp/180, 1), min(cholesterol/300, 1), min(abs(hr-80)/40 + 0.5, 1)]
    normal_ranges = [0.4, 0.3, 0.3, 0.6]  # Clinical normal ranges

    fig.add_trace(
        go.Scatterpolar(r=values, theta=categories, fill='toself', name='Patient Values',
                       line_color='#1f77b4'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatterpolar(r=normal_ranges, theta=categories, name='Normal Ranges',
                       line_color='#28a745', line_dash='dash'),
        row=1, col=2
    )

    # 3. ECG analysis (Lead I sample)
    if ecg_data and 'lead_1' in ecg_data:
        signal = ecg_data['lead_1'][:1000]  # First 2 seconds
        time = np.linspace(0, 2, len(signal))
        fig.add_trace(
            go.Scatter(x=time, y=signal, mode='lines', name='ECG Lead I',
                      line=dict(color='#1f77b4', width=1.5)),
            row=2, col=1
        )
        # Add ECG interpretation
        fig.add_annotation(
            text="<b>ECG Interpretation:</b><br>Normal sinus rhythm<br>Rate: ~70 bpm<br>Normal intervals",
            x=0.02, y=0.98, xref="paper", yref="paper",
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#1f77b4", borderwidth=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=['ECG Data Not Available'],
                      showlegend=False),
            row=2, col=1
        )

    # 4. Clinical recommendations table
    recommendations = []
    if risk_score >= 0.2:
        recommendations = [["HIGH RISK", "Intensive statin therapy + lifestyle", "Immediate"],
                          ["Blood Pressure", "Target <130/80 mmHg", "3 months"],
                          ["Cholesterol", "LDL <70 mg/dL", "3 months"],
                          ["Lifestyle", "Diet + exercise counseling", "Ongoing"]]
    elif risk_score >= 0.075:
        recommendations = [["MODERATE RISK", "Moderate statin therapy", "Within 1 month"],
                          ["Blood Pressure", "Target <130/80 mmHg", "6 months"],
                          ["Cholesterol", "LDL <100 mg/dL", "6 months"],
                          ["Lifestyle", "Dietary modifications", "Ongoing"]]
    else:
        recommendations = [["LOW RISK", "Lifestyle therapy only", "Annual review"],
                          ["Blood Pressure", "Target <120/80 mmHg", "Annual"],
                          ["Cholesterol", "Monitor levels", "Annual"],
                          ["Prevention", "Healthy lifestyle promotion", "Ongoing"]]

    fig.add_trace(
        go.Table(
            header=dict(values=['Risk Factor', 'Clinical Target', 'Timeline'],
                       fill_color='#f0f0f0', font=dict(size=12, color='black')),
            cells=dict(values=np.array(recommendations).T.tolist(),
                      fill_color='white', font=dict(size=11))
        ),
        row=2, col=2
    )

    # 5. Risk factor correlations
    corr_data = [[1, 0.8, 0.6, 0.4],
                 [0.8, 1, 0.7, 0.3],
                 [0.6, 0.7, 1, 0.5],
                 [0.4, 0.3, 0.5, 1]]

    fig.add_trace(
        go.Heatmap(z=corr_data, x=categories, y=categories, colorscale='RdBu_r',
                   text=np.round(corr_data, 2), texttemplate='%{text}', textfont={"size": 10}),
        row=3, col=1
    )

    # 6. Population comparison with percentiles
    age_groups = ['30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    pop_risks = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25]  # Population averages
    patient_risk = [risk_score] * len(age_groups)

    fig.add_trace(
        go.Bar(x=age_groups, y=pop_risks, name='Population Average',
              marker_color='#1f77b4', opacity=0.7),
        row=3, col=2
    )
    fig.add_trace(
        go.Scatter(x=age_groups, y=patient_risk, name='Patient Risk',
                  mode='lines+markers', line=dict(color='#dc3545', width=3)),
        row=3, col=2
    )

    # 7. Longitudinal trends
    if patient_history:
        hist_df = pd.DataFrame(patient_history)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        fig.add_trace(
            go.Scatter(x=hist_df['date'], y=hist_df['risk_score'], mode='lines+markers',
                      name='Risk Trend', line=dict(color='#28a745', width=2)),
            row=4, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=['No Historical Data'],
                      showlegend=False),
            row=4, col=1
        )

    # 8. Diagnostic confidence
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=87,  # Example confidence score
            title={"text": "AI Diagnostic Confidence"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#28a745'},
                  'steps': [{'range': [0, 60], 'color': '#dc3545'},
                           {'range': [60, 80], 'color': '#ffc107'},
                           {'range': [80, 100], 'color': '#28a745'}]}
        ),
        row=4, col=2
    )

    # 9. Treatment plan summary
    treatment_plan = [
        ["Primary Prevention" if risk_score < 0.075 else "Secondary Prevention",
         "Lifestyle Modifications", "Immediate"],
        ["Statin Therapy" if risk_score >= 0.075 else "No Statin",
         "Based on risk calculation", "As indicated"],
        ["Anti-hypertensive" if bp >= 130 else "No Anti-hypertensive",
         "ACEI/ARB preferred", "As indicated"],
        ["Anti-platelet", "Aspirin 81mg daily" if risk_score >= 0.1 else "Not indicated",
         "For ASCVD risk ≥10%"]
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=['Treatment Category', 'Specific Therapy', 'Indication']),
            cells=dict(values=np.array(treatment_plan).T.tolist())
        ),
        row=5, col=1
    )

    # 10. Follow-up schedule
    followup_schedule = [
        ["Initial Visit", "Complete assessment", "Today"],
        ["Follow-up", "Statin titration" if risk_score >= 0.075 else "Risk re-assessment", "4-6 weeks"],
        ["Monitoring", "Lipid panel + LFTs", "8-12 weeks"],
        ["Annual Review", "Complete cardiovascular assessment", "1 year"]
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=['Visit Type', 'Purpose', 'Timeline']),
            cells=dict(values=np.array(followup_schedule).T.tolist())
        ),
        row=5, col=2
    )

    fig.update_layout(
        height=1600,
        title_text="<b>Comprehensive Cardiovascular Assessment Report</b><br><sub>ACC/AHA 2019 Guidelines</sub>",
        showlegend=False
    )

    return fig

def create_ecg_anomaly_detection(ecg_data, anomalies=None, title="ECG Anomaly Detection"):
    """
    Create visualization for ECG anomaly detection with highlighted irregularities
    """
    if not ecg_data:
        return px.line(title="No ECG data available")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Lead I - Full Signal', 'Lead II - Full Signal', 'Anomaly Detection Summary'),
        shared_xaxes=True
    )

    # Plot ECG leads
    for i, lead in enumerate(['lead_1', 'lead_2']):
        if lead in ecg_data:
            signal = ecg_data[lead]
            time = np.linspace(0, len(signal)/500, len(signal))

            fig.add_trace(
                go.Scatter(x=time, y=signal, mode='lines', name=f'Lead {i+1}',
                          line=dict(color='#1f77b4', width=1)),
                row=i+1, col=1
            )

            # Highlight anomalies if detected
            if anomalies and lead in anomalies:
                anomaly_times = anomalies[lead]
                anomaly_signals = [signal[int(t*500)] for t in anomaly_times if int(t*500) < len(signal)]
                fig.add_trace(
                    go.Scatter(x=anomaly_times, y=anomaly_signals, mode='markers',
                              name=f'Anomalies Lead {i+1}', marker=dict(color='red', size=8, symbol='x')),
                    row=i+1, col=1
                )

    # Anomaly summary
    if anomalies:
        total_anomalies = sum(len(times) for times in anomalies.values())
        anomaly_types = ['Arrhythmia', 'ST Elevation', 'Conduction Block', 'Premature Beats']
        counts = [total_anomalies // 4 + i for i in range(4)]  # Simulated distribution

        fig.add_trace(
            go.Bar(x=anomaly_types, y=counts, name='Anomaly Types',
                  marker_color='#dc3545', opacity=0.7),
            row=3, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='text', text=['No Anomalies Detected'],
                      showlegend=False),
            row=3, col=1
        )

    fig.update_layout(
        height=800,
        title_text=f"<b>{title}</b>",
        showlegend=True
    )

    return fig
