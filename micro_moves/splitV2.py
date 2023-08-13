import h5py

def split_dataset(file_path):
    # Define the output file paths
    output_paths = {
        "bright": file_path.replace("cropped_ions.h5", "bright_ions.h5"),
        "dark": file_path.replace("cropped_ions.h5", "dark_ions.h5"),
        "halfpi": file_path.replace("cropped_ions.h5", "halfpi_ions.h5")
    }

    with h5py.File(file_path, "r") as original_file:
        # Iterate through the keys and copy data into separate files
        for category in ["bright", "dark", "halfpi"]:
            with h5py.File(output_paths[category], "w") as output_file:
                for key in original_file.keys():
                    if category in key:
                        output_file.create_dataset(key, data=original_file[key])
            print(f"{category} data has been written to {output_paths[category]}")

    print("Dataset split successfully!")
    return output_paths


def check_split_success(file_path, output_paths):
    with h5py.File(file_path, "r") as original_file:
        original_keys = set(original_file.keys())

        for category, output_path in output_paths.items():
            with h5py.File(output_path, "r") as output_file:
                output_keys = set(output_file.keys())
                for key in output_keys:
                    if category not in key or key not in original_keys:
                        print(f"Error: Key {key} in {output_path} is incorrect!")
                        return False

                print(f"{category} file keys are correct!")

    print("All files have been successfully verified!")
    return True


file_path = "C:/Users/Seiven/Desktop/MY_MLmodels/ions2/binary/cropped_ions.h5"

# Split the original dataset
output_paths = split_dataset(file_path)

# Check if the split was successful
success = check_split_success(file_path, output_paths)
if success:
    print("Data has been successfully split and verified!")
else:
    print("An error occurred during verification.")
