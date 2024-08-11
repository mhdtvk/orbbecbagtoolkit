from test_data_extractor import ExtractData
import subprocess
import yaml
import os

# This class manages directory creation for saving files
class CreateDirectory:
    def __init__(self) -> None:
        pass

    def get_parent_dirs(self, path, levels=2):
        """Generates a list of parent directories up to a specified level."""
        directory_lvl_name = []
        base_name = os.path.splitext(os.path.basename(path))[0]
        directory_lvl_name.append(base_name)
        for _ in range(levels):
            path = os.path.dirname(path)
            directory_lvl_name.append(os.path.basename(path))
        directory_lvl_name.append('ir_depth_raw_files')
        directory_lvl_name.reverse()
        return directory_lvl_name

    def create_output_directory(self, input_path):
        """Creates the output directory based on the input path."""
        folders_name = self.get_parent_dirs(input_path)
        output_path = os.path.join(os.getcwd(), *folders_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

# This class navigates through the camera bag files and processes them
class CameraBagNavigator:
    def __init__(self) -> None:
        self.create_directory = CreateDirectory()
        self.data_extractor = ExtractData()
        self.camera_list = []

    # Extracting the frame number from the Master Camera to use it as a reference frame-number
    def get_number_messages(self, bag_input_path):
        """Extracts the number of messages from the master camera bag file."""
        info_dict = yaml.load(
            subprocess.Popen(['rosbag', 'info', '--yaml', bag_input_path], stdout=subprocess.PIPE).communicate()[0],
            Loader=yaml.FullLoader
        )
        max_message_number = 0
        for topic in info_dict['topics']:
            max_message_number = max(max_message_number, topic['messages'])
        return max_message_number

    # Running the data extractor class for each camera bag file
    def process_bagfiles_in_dir(self, input_directory, master_camera_name, convert_depth_to_8bit, convert_ir_to_8bit, policy, batch_size):
        """Processes all bag files in the directory."""
        self.data_extractor.convert_depth_to_8bit = convert_depth_to_8bit
        self.data_extractor.convert_ir_to_8bit = convert_ir_to_8bit
        self.data_extractor.policy = policy

        # Extracting frame numbers based on the master camera frame number
        master_camera_file = master_camera_name + '.bag'
        master_camera_path = os.path.join(input_directory, master_camera_file)
        frame_number = self.get_number_messages(master_camera_path)
        print(f"\n[INFO] The Number of Frames for Master Camera : ",{frame_number})
        output_path = self.create_directory.create_output_directory(input_directory)

        # Creating the list of cameras
        for filename in os.listdir(input_directory):
            file_path = os.path.join(input_directory, filename)
            if os.path.isfile(file_path) and file_path.endswith('.bag'):
                cam_name, _ = os.path.splitext(filename)
                camera_path = [cam_name, file_path]
                self.camera_list.append(camera_path)

        # Processing frames in batches
        for last_frame in range(0, frame_number, batch_size):
            if last_frame:
                start_frame = last_frame - 50
                self.extract_cycle(start_frame, last_frame, output_path)
                self.data_extractor.save_datasets(start_frame)
        if last_frame < frame_number:
            rest_frames = frame_number - last_frame
            self.extract_cycle(start_frame=last_frame, last_frame=(last_frame + rest_frames), output_path=output_path)
            self.data_extractor.save_datasets(last_frame)

    # Frame processing and storage cycle
    def extract_cycle(self, start_frame, last_frame, output_path):
        """Extracts and processes frames for each camera."""
        print(f'\n[INFO] Processing frames {start_frame} to {last_frame} :')
        for cam in self.camera_list:
            self.data_extractor.extract_frames(cam[1], output_path, cam[0], start_frame, last_frame)
