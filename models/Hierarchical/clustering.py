from models.utils.feature_engineering import load_data

def main():
    data = load_data("../../data")
    print(f"number of features extracted: {len(data.columns)}")

if __name__ == "__main__":
    main()