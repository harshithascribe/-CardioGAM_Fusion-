from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime, timedelta
import bcrypt
import secrets

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(50), default='user')  # admin, doctor, nurse
    is_verified = db.Column(db.Boolean, default=False)
    verification_token = db.Column(db.String(100), unique=True)
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expires = db.Column(db.DateTime)
    otp_code = db.Column(db.String(6))
    otp_expires = db.Column(db.DateTime)
    google_id = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    def generate_verification_token(self):
        self.verification_token = secrets.token_urlsafe(32)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expires = datetime.utcnow() + timedelta(hours=1)

    def generate_otp(self):
        import random
        self.otp_code = f"{random.randint(100000, 999999)}"
        self.otp_expires = datetime.utcnow() + timedelta(minutes=10)

    def verify_otp(self, otp):
        if self.otp_code == otp and datetime.utcnow() < self.otp_expires:
            self.otp_code = None
            self.otp_expires = None
            return True
        return False

    def verify_reset_token(self, token):
        return self.reset_token == token and datetime.utcnow() < self.reset_token_expires

class PatientAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)
    risk_score = db.Column(db.Float, nullable=False)
    risk_category = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    recommendations = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    # ECG features
    lead_1_mean = db.Column(db.Float)
    lead_1_std = db.Column(db.Float)
    lead_2_mean = db.Column(db.Float)
    lead_2_std = db.Column(db.Float)
    lead_3_mean = db.Column(db.Float)
    lead_3_std = db.Column(db.Float)
    lead_4_mean = db.Column(db.Float)
    lead_4_std = db.Column(db.Float)
    lead_5_mean = db.Column(db.Float)
    lead_5_std = db.Column(db.Float)
    lead_6_mean = db.Column(db.Float)
    lead_6_std = db.Column(db.Float)
    lead_7_mean = db.Column(db.Float)
    lead_7_std = db.Column(db.Float)
    lead_8_mean = db.Column(db.Float)
    lead_8_std = db.Column(db.Float)
    lead_9_mean = db.Column(db.Float)
    lead_9_std = db.Column(db.Float)
    lead_10_mean = db.Column(db.Float)
    lead_10_std = db.Column(db.Float)
    lead_11_mean = db.Column(db.Float)
    lead_11_std = db.Column(db.Float)
    lead_12_mean = db.Column(db.Float)
    lead_12_std = db.Column(db.Float)

    def to_dict(self):
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'age': self.age,
            'bp': self.bp,
            'cholesterol': self.cholesterol,
            'heart_rate': self.heart_rate,
            'risk_score': self.risk_score,
            'risk_category': self.risk_category,
            'confidence': self.confidence,
            'recommendations': self.recommendations,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }
