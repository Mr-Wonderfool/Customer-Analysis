from models.utils.feature_engineering import load_data, standard_scale

def main():
    data = load_data("../../data")
    print(f"number of features extracted: {len(data.columns)}")
    scaled_data = standard_scale(data)

if __name__ == "__main__":
    main()