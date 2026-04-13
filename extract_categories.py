import pickle
import sys

def main():
    try:
        with open('encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
        
        # Check if it's an OrdinalEncoder, OneHotEncoder, etc.
        if hasattr(encoder, 'categories_'):
            for i, cat in enumerate(encoder.categories_):
                print(f"Feature {i}: {cat}")
        elif type(encoder).__name__ == 'ColumnTransformer':
            print("It's a ColumnTransformer. Transformers:")
            for name, trans, columns in encoder.transformers_:
                if hasattr(trans, 'categories_'):
                    print(f"Categories for {columns}: {trans.categories_}")
        else:
            print("Unknown encoder type:", type(encoder))
            print(dir(encoder))

    except Exception as e:
        print(f"Error loading encoder: {e}")

if __name__ == '__main__':
    main()
