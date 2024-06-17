from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class StockPredictionForm(FlaskForm):
  ticker_symbol = StringField('Stock Ticker Symbol:', validators=[DataRequired()])
  submit = SubmitField('Predict Stock')
