import pickle
import pandas as pd
import re

def main():
    try:
        # Load components
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoder.pkl', 'rb') as f:
            encoders = pickle.load(f)

        data = {
            'property_type': 'house',
            'areaWithType': 'Built Up area: 1210 (112.41 sq.m.)',
            'additionalRoom': 'not available',
            'facing': 'East',
            'agePossession': '0 to 1 Year Old',
            'bedRoom': 2,
            'bathroom': 2,
            'balcony': 1,
            'floorNum': 1
        }
        
        # Calculate area
        match = re.search(r'\d+', data['areaWithType'])
        area = float(match.group()) if match else 1000.0
        
        # Add missing fields
        data['area'] = area
        data['price_per_sqft'] = 8000.0 # Dummy value
        
        # Build dataframe in the EXACT order scaler expects
        expected_cols = ["property_type", "price_per_sqft", "area", "areaWithType", 
                         "bedRoom", "bathroom", "balcony", "additionalRoom", 
                         "floorNum", "facing", "agePossession"]
        
        df = pd.DataFrame([data], columns=expected_cols)
        
        # Apply encoders
        for col, en in encoders.items():
            if col in df.columns:
                df[col] = en.transform(df[col])
                
        # Apply scaler
        scaled_df = scaler.transform(df)
        
        # Predict
        pred = model.predict(scaled_df)
        print("Prediction with 8000:", pred)
        
        # Try another price_per_sqft
        df.loc[0, 'price_per_sqft'] = 15000.0
        scaled_df2 = scaler.transform(df)
        pred2 = model.predict(scaled_df2)
        print("Prediction with 15000:", pred2)

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
