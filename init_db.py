#!/usr/bin/env python3
"""
Initialize the database for testing purposes
"""
import sys
import os
sys.path.insert(0, '.')

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from src.dashboard.models import db, User, PatientAssessment

def init_database():
    """Initialize the database without starting the server"""
    # Create Flask app with same config as main app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cardio_fusion.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)

    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")

        # Create default users if not exists
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            print("✅ Default admin user created")

        if not User.query.filter_by(username='doctor').first():
            doctor = User(username='doctor', role='doctor')
            doctor.set_password('doctor123')
            db.session.add(doctor)
            print("✅ Default doctor user created")

        if not User.query.filter_by(username='nurse').first():
            nurse = User(username='nurse', role='nurse')
            nurse.set_password('nurse123')
            db.session.add(nurse)
            print("✅ Default nurse user created")

        db.session.commit()

        # Check table counts
        users_count = User.query.count()
        patients_count = PatientAssessment.query.count()
        print(f"📊 Users table: {users_count} records")
        print(f"📊 Patient assessments table: {patients_count} records")

if __name__ == "__main__":
    print("🔧 Initializing CardioGAM-Fusion Database...")
    init_database()
    print("✅ Database initialization completed")
