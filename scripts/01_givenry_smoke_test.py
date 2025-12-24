import os

def main():
    # Put your token in PyCharm Run Config env vars as:
    # GIVERNY_AUTH_TOKEN=xxxxx
    token = os.getenv("GIVERNY_AUTH_TOKEN", "edu.jhu.pha.turbulence.testing-201406")

    from givernylocal.turbulence_dataset import turb_dataset

    dataset_title = "channel"
    output_path = "./giverny_output"

    ds = turb_dataset(dataset_title=dataset_title, output_path=output_path, auth_token=token)

    print("✅ givernylocal dataset instantiated")
    print("Dataset title:", dataset_title)
    print("Output path:", output_path)

if __name__ == "__main__":
    main()
