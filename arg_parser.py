try:
    from ir_depth_extractor import CameraBagNavigator
    import argparse
except ImportError as e:
    print(f"Error: {e}")
    exit(1)

def main():
    """
    Main function to parse command-line arguments to run the Data Extractor.

    This function parses command-line arguments to configure and run the Data Extractor.
    """
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-f', '--folder_path', type=str, required=True, help='Path of the root folder containing recorded sensor files.')
        parser.add_argument('-m', '--master_camera', type=str, required=False, help='The name of the master camera.', default='camera_0')
        parser.add_argument('-cd', '--convert_depth_to_8bit', type=bool, required=False, help='Convert depth frames to 8-bit format.', default=False)
        parser.add_argument('-ci', '--convert_ir_to_8bit', type=bool, required=False, help='Convert IR frames to 8-bit format.', default=False)
        parser.add_argument('-p', '--policy', type=str, required=False, help='Policy to resolve abnormalities.', default='drop')
        parser.add_argument('-b', '--batch_size', type=int, required=False, help='Number of frames to process in each batch.', default=50)
        
        args = parser.parse_args()
        folder_path = args.folder_path
        master_camera = args.master_camera
        convert_depth_to_8bit = args.convert_depth_to_8bit
        convert_ir_to_8bit = args.convert_ir_to_8bit
        policy = args.policy
        batch_size = args.batch_size

        # Initialize and run the data extractor
        data_extractor = CameraBagNavigator()
        data_extractor.process_bagfiles_in_dir(folder_path, master_camera, convert_depth_to_8bit, convert_ir_to_8bit, policy, batch_size)

    except ImportError as error:
        print(f"Error: {error}")
        exit(1)

if __name__ == "__main__":
    main()
