"""
Advanced search and filtering utilities for CardioGAM-Fusion++ Dashboard
"""
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, func
from src.dashboard.models import PatientAssessment, User

class PatientSearchEngine:
    """Advanced search engine for patient data"""

    def __init__(self):
        self.filters = {}

    def add_filter(self, field, operator, value):
        """Add a search filter"""
        self.filters[field] = {'operator': operator, 'value': value}
        return self

    def clear_filters(self):
        """Clear all filters"""
        self.filters = {}
        return self

    def search(self, query_string="", limit=100, offset=0):
        """Execute search with filters"""
        base_query = PatientAssessment.query

        # Apply text search
        if query_string:
            search_filter = or_(
                PatientAssessment.patient_id.ilike(f"%{query_string}%"),
                PatientAssessment.risk_category.ilike(f"%{query_string}%"),
                PatientAssessment.recommendations.ilike(f"%{query_string}%")
            )
            base_query = base_query.filter(search_filter)

        # Apply field filters
        for field, filter_info in self.filters.items():
            operator = filter_info['operator']
            value = filter_info['value']

            if field == 'age':
                if operator == 'eq':
                    base_query = base_query.filter(PatientAssessment.age == value)
                elif operator == 'gt':
                    base_query = base_query.filter(PatientAssessment.age > value)
                elif operator == 'lt':
                    base_query = base_query.filter(PatientAssessment.age < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.age.between(value[0], value[1]))

            elif field == 'bp':
                if operator == 'eq':
                    base_query = base_query.filter(PatientAssessment.bp == value)
                elif operator == 'gt':
                    base_query = base_query.filter(PatientAssessment.bp > value)
                elif operator == 'lt':
                    base_query = base_query.filter(PatientAssessment.bp < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.bp.between(value[0], value[1]))

            elif field == 'cholesterol':
                if operator == 'eq':
                    base_query = base_query.filter(PatientAssessment.cholesterol == value)
                elif operator == 'gt':
                    base_query = base_query.filter(PatientAssessment.cholesterol > value)
                elif operator == 'lt':
                    base_query = base_query.filter(PatientAssessment.cholesterol < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.cholesterol.between(value[0], value[1]))

            elif field == 'heart_rate':
                if operator == 'eq':
                    base_query = base_query.filter(PatientAssessment.heart_rate == value)
                elif operator == 'gt':
                    base_query = base_query.filter(PatientAssessment.heart_rate > value)
                elif operator == 'lt':
                    base_query = base_query.filter(PatientAssessment.heart_rate < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.heart_rate.between(value[0], value[1]))

            elif field == 'risk_score':
                if operator == 'eq':
                    base_query = base_query.filter(PatientAssessment.risk_score == value)
                elif operator == 'gt':
                    base_query = base_query.filter(PatientAssessment.risk_score > value)
                elif operator == 'lt':
                    base_query = base_query.filter(PatientAssessment.risk_score < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.risk_score.between(value[0], value[1]))

            elif field == 'risk_category':
                base_query = base_query.filter(PatientAssessment.risk_category.ilike(f"%{value}%"))

            elif field == 'created_at':
                if operator == 'after':
                    base_query = base_query.filter(PatientAssessment.created_at > value)
                elif operator == 'before':
                    base_query = base_query.filter(PatientAssessment.created_at < value)
                elif operator == 'between':
                    base_query = base_query.filter(PatientAssessment.created_at.between(value[0], value[1]))

        # Apply pagination
        total_count = base_query.count()
        results = base_query.order_by(PatientAssessment.created_at.desc()).offset(offset).limit(limit).all()

        return {
            'results': [r.to_dict() for r in results],
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        }

class AnalyticsFilterEngine:
    """Advanced filtering for analytics data"""

    def __init__(self):
        self.date_range = None
        self.risk_categories = []
        self.age_range = None
        self.bp_range = None
        self.cholesterol_range = None
        self.heart_rate_range = None

    def set_date_range(self, start_date, end_date):
        """Set date range filter"""
        self.date_range = (start_date, end_date)
        return self

    def set_risk_categories(self, categories):
        """Set risk category filters"""
        self.risk_categories = categories
        return self

    def set_age_range(self, min_age, max_age):
        """Set age range filter"""
        self.age_range = (min_age, max_age)
        return self

    def set_bp_range(self, min_bp, max_bp):
        """Set blood pressure range filter"""
        self.bp_range = (min_bp, max_bp)
        return self

    def set_cholesterol_range(self, min_chol, max_chol):
        """Set cholesterol range filter"""
        self.cholesterol_range = (min_chol, max_chol)
        return self

    def set_heart_rate_range(self, min_hr, max_hr):
        """Set heart rate range filter"""
        self.heart_rate_range = (min_hr, max_hr)
        return self

    def apply_filters(self, dataframe):
        """Apply all filters to a pandas DataFrame"""
        df = dataframe.copy()

        # Date range filter
        if self.date_range:
            start_date, end_date = self.date_range
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df = df[(df['created_at'] >= start_date) & (df['created_at'] <= end_date)]

        # Risk category filter
        if self.risk_categories:
            if 'risk_category' in df.columns:
                df = df[df['risk_category'].isin(self.risk_categories)]

        # Age range filter
        if self.age_range:
            min_age, max_age = self.age_range
            if 'age' in df.columns:
                df = df[(df['age'] >= min_age) & (df['age'] <= max_age)]

        # Blood pressure range filter
        if self.bp_range:
            min_bp, max_bp = self.bp_range
            if 'bp' in df.columns:
                df = df[(df['bp'] >= min_bp) & (df['bp'] <= max_bp)]

        # Cholesterol range filter
        if self.cholesterol_range:
            min_chol, max_chol = self.cholesterol_range
            if 'cholesterol' in df.columns:
                df = df[(df['cholesterol'] >= min_chol) & (df['cholesterol'] <= max_chol)]

        # Heart rate range filter
        if self.heart_rate_range:
            min_hr, max_hr = self.heart_rate_range
            if 'heart_rate' in df.columns:
                df = df[(df['heart_rate'] >= min_hr) & (df['heart_rate'] <= max_hr)]

        return df

    def get_filter_summary(self):
        """Get a summary of applied filters"""
        summary = {}

        if self.date_range:
            summary['Date Range'] = f"{self.date_range[0]} to {self.date_range[1]}"

        if self.risk_categories:
            summary['Risk Categories'] = ", ".join(self.risk_categories)

        if self.age_range:
            summary['Age Range'] = f"{self.age_range[0]} - {self.age_range[1]} years"

        if self.bp_range:
            summary['Blood Pressure'] = f"{self.bp_range[0]} - {self.bp_range[1]} mmHg"

        if self.cholesterol_range:
            summary['Cholesterol'] = f"{self.cholesterol_range[0]} - {self.cholesterol_range[1]} mg/dL"

        if self.heart_rate_range:
            summary['Heart Rate'] = f"{self.heart_rate_range[0]} - {self.heart_rate_range[1]} bpm"

        return summary

def create_advanced_search_form():
    """Create an advanced search form component"""
    import dash_bootstrap_components as dbc
    from dash import html, dcc

    form = dbc.Card([
        dbc.CardHeader("Advanced Search & Filters"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Search Query"),
                    dbc.Input(id="search-query", type="text", placeholder="Search patient ID, risk category, or recommendations"),
                ], width=12),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Age Range"),
                    dcc.RangeSlider(
                        id="age-range-filter",
                        min=18, max=100, step=1,
                        value=[18, 100],
                        marks={18: '18', 40: '40', 60: '60', 80: '80', 100: '100'}
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Risk Score Range"),
                    dcc.RangeSlider(
                        id="risk-score-filter",
                        min=0, max=1, step=0.01,
                        value=[0, 1],
                        marks={0: '0.0', 0.2: '0.2', 0.4: '0.4', 0.6: '0.6', 0.8: '0.8', 1.0: '1.0'}
                    ),
                ], width=6),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Blood Pressure Range (mmHg)"),
                    dcc.RangeSlider(
                        id="bp-range-filter",
                        min=80, max=200, step=5,
                        value=[80, 200],
                        marks={80: '80', 120: '120', 160: '160', 200: '200'}
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Cholesterol Range (mg/dL)"),
                    dcc.RangeSlider(
                        id="cholesterol-range-filter",
                        min=100, max=400, step=10,
                        value=[100, 400],
                        marks={100: '100', 200: '200', 300: '300', 400: '400'}
                    ),
                ], width=6),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Date Range"),
                    dcc.DatePickerRange(
                        id="date-range-filter",
                        start_date=(datetime.now() - timedelta(days=90)).date(),
                        end_date=datetime.now().date(),
                        display_format='YYYY-MM-DD'
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Risk Categories"),
                    dcc.Checklist(
                        id="risk-category-filter",
                        options=[
                            {"label": "Low Risk", "value": "Low Risk"},
                            {"label": "Moderate Risk", "value": "Moderate Risk"},
                            {"label": "High Risk", "value": "High Risk"}
                        ],
                        value=["Low Risk", "Moderate Risk", "High Risk"],
                        inline=True
                    ),
                ], width=6),
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Search", id="advanced-search-btn", color="primary", className="me-2"),
                    dbc.Button("Clear Filters", id="clear-filters-btn", color="secondary"),
                ], width=12),
            ]),
            html.Br(),
            html.Div(id="search-results-summary")
        ])
    ])

    return form

def create_export_controls():
    """Create export control components"""
    import dash_bootstrap_components as dbc
    from dash import html, dcc

    controls = dbc.Card([
        dbc.CardHeader("Export Options"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Export Format"),
                    dcc.Dropdown(
                        id="export-format",
                        options=[
                            {"label": "CSV", "value": "csv"},
                            {"label": "Excel", "value": "excel"},
                            {"label": "PDF Report", "value": "pdf"}
                        ],
                        value="csv",
                        clearable=False
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Label("Data Scope"),
                    dcc.Dropdown(
                        id="export-scope",
                        options=[
                            {"label": "Current Search Results", "value": "search"},
                            {"label": "All Filtered Data", "value": "filtered"},
                            {"label": "All Patient Data", "value": "all"}
                        ],
                        value="search",
                        clearable=False
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Button("Export Data", id="export-btn", color="success", className="mt-4"),
                    dcc.Download(id="download-data")
                ], width=4),
            ])
        ])
    ])

    return controls
