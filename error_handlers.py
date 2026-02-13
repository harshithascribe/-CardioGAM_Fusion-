"""
Error handling utilities for CardioGAM-Fusion++ Dashboard
"""
import traceback
import logging
from flask import current_app, flash, redirect, url_for, render_template_string
from dash import html, dcc
import dash_bootstrap_components as dbc

# Set up logger
logger = logging.getLogger(__name__)

class CardioFusionError(Exception):
    """Base exception class for CardioGAM-Fusion++"""
    pass

class ValidationError(CardioFusionError):
    """Validation error for input data"""
    pass

class ModelError(CardioFusionError):
    """Error related to ML model operations"""
    pass

class DatabaseError(CardioFusionError):
    """Database operation error"""
    pass

def handle_validation_errors(func):
    """Decorator to handle validation errors in callbacks"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error: {str(e)}")
            return create_error_alert(f"Validation Error: {str(e)}", "warning")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return create_error_alert("An unexpected error occurred. Please try again.", "danger")
    return wrapper

def create_error_alert(message, color="danger"):
    """Create a Bootstrap alert component for errors"""
    return dbc.Alert([
        html.I(className="fas fa-exclamation-triangle me-2"),
        html.Strong("Error: "),
        message
    ], color=color, dismissable=True, className="mb-3")

def create_success_alert(message):
    """Create a Bootstrap alert component for success messages"""
    return dbc.Alert([
        html.I(className="fas fa-check-circle me-2"),
        message
    ], color="success", dismissable=True, className="mb-3")

def create_warning_alert(message):
    """Create a Bootstrap alert component for warnings"""
    return dbc.Alert([
        html.I(className="fas fa-exclamation-circle me-2"),
        message
    ], color="warning", dismissable=True, className="mb-3")

def validate_patient_data(age, bp, cholesterol, heart_rate):
    """Validate patient input data"""
    errors = []

    # Age validation
    if not isinstance(age, (int, float)) or age < 18 or age > 100:
        errors.append("Age must be between 18 and 100 years")

    # Blood pressure validation
    if not isinstance(bp, (int, float)) or bp < 80 or bp > 200:
        errors.append("Blood pressure must be between 80 and 200 mmHg")

    # Cholesterol validation
    if not isinstance(cholesterol, (int, float)) or cholesterol < 100 or cholesterol > 400:
        errors.append("Cholesterol must be between 100 and 400 mg/dL")

    # Heart rate validation
    if not isinstance(heart_rate, (int, float)) or heart_rate < 40 or heart_rate > 150:
        errors.append("Heart rate must be between 40 and 150 bpm")

    if errors:
        return {"valid": False, "message": "; ".join(errors)}

    return {"valid": True, "message": ""}

def handle_model_loading_error(model_name, error):
    """Handle model loading errors gracefully"""
    logger.error(f"Failed to load {model_name}: {str(error)}")
    return create_error_alert(f"Failed to load {model_name}. Some features may not work properly.", "warning")

def handle_database_error(operation, error):
    """Handle database operation errors"""
    logger.error(f"Database error during {operation}: {str(error)}")
    return create_error_alert("Database error occurred. Please try again later.", "danger")

def create_error_boundary(component, error_message="Something went wrong"):
    """Create an error boundary wrapper for Dash components"""
    try:
        return component
    except Exception as e:
        logger.error(f"Error rendering component: {str(e)}")
        return dbc.Alert([
            html.I(className="fas fa-bug me-2"),
            html.Strong("Component Error: "),
            error_message
        ], color="danger", className="m-3")

def log_user_action(user_id, action, details=None):
    """Log user actions for audit purposes"""
    logger.info(f"User {user_id}: {action}" + (f" - {details}" if details else ""))

def create_loading_spinner():
    """Create a loading spinner component"""
    return dbc.Spinner(color="primary", type="grow", children=[
        html.Span("Loading...", className="visually-hidden")
    ])
