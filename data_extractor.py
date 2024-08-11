import os
import numpy as np
import cv2
from plyfile import PlyData, PlyElement
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBError
from utils import frame_to_bgr_image

class ExtractData:
    def __init__(self) -> None:
        self.result_dataset = {}
        self.input_path: str = None
        self.output_path = None
        self.convert_depth_to_8bit = True
        self.convert_ir_to_8bit = True
        self.policy = 'copy'
        self.drop_list = []  # List of frames to drop

    def create_result_datasets(self, cam_name):
        """Initialize the result dataset structure for a camera."""
        self.result_dataset[cam_name] = {
            'ir_frames': {
                'removed_frms': [],
                'abnormal_tms_index': [],
                'timestamps': [],
                'copy': [],
                'drop': []
            },
            'depth_frames': {
                'removed_frms': [],
                'abnormal_tms_index': [],
                'timestamps': [],
                'copy': [],
                'drop': []
            }
        }

    def save_datasets(self, start_frame):
        """Save datasets after processing frames."""
        self.tmscheck(self.policy)
        if self.policy == 'drop':
            self.drop_frmsets()

        if not self.cardinality_check(self.policy):
            print('\n[ERROR] Cardinality check failed.')
            exit(0)
        self.save_tms_file()
        if self.save_frames(start_frame):
            print("[INFO] Frames saved!")
        else:
            print("[ERROR] Frames Not saved!")

        self.generate_metadata()
        self.result_dataset = {}

    def save_frames(self, start_frame):
        """Save depth and IR frames."""
        try:
            self.save_depth_frame(start_frame)
            self.save_ir_frame(start_frame)
            return True
        except Exception:
            return False

    def extract_frames(self, input_path, output_path, cam_name, start_frame, last_frame):
        """Extract frames from the input path and save them to the output path."""
        self.output_path = output_path
        self.input_path = input_path
        self.create_result_datasets(cam_name)
        self.create_output_dir(output_path=output_path)
        pipeline = Pipeline(self.input_path)
        playback = pipeline.get_playback()
        config = Config()

        # Save device info
        self.get_device_info(
            output_path, cam_name, device_info=str(playback.get_device_info()),
            camera_param=str(pipeline.get_camera_param())
        )
        # Configure pipeline
        try:
            depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            if depth_profile_list:
                depth_profile = depth_profile_list.get_default_video_stream_profile()
                config.enable_stream(depth_profile)

            ir_profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
            if ir_profile_list:
                ir_profile = ir_profile_list.get_default_video_stream_profile()
                config.enable_stream(ir_profile)

            color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if color_profile_list:
                color_profile = color_profile_list.get_default_video_stream_profile()
                config.enable_stream(color_profile)
                has_color_sensor = True
            pipeline.start(config)
        except OBError as e:
            print(e)
            return

        # Initialize the arguments
        frame_index = start_frame - 1
        # Data acquisition loop
        try:
            while True:
                if self.policy == 'copy':
                    frames = pipeline.wait_for_frames(100)

                    frame_index += 1
                    if frame_index >= last_frame:
                        break
                    elif frames is None:
                        print(f'\n[ERROR]: Frameset Lost for Frame index: {frame_index} {cam_name}\n')
                        frames = self.replacment_frmset(frame_index, cam_name, replace_tms=True)
                    elif frames.get_depth_frame() is None or frames.get_ir_frame() is None:
                        print(f'\n[ERROR]: IR or Depth Frame Lost for Frame index: {frame_index} {cam_name}\n')
                        frames = self.replacment_frmset(frame_index=frame_index, camera_name=cam_name, replace_tms=True)
                    else:
                        depth_frame = frames.get_depth_frame()
                        ir_frame = frames.get_ir_frame()

                        self.result_dataset[cam_name]['ir_frames']['copy'].append(ir_frame)
                        self.result_dataset[cam_name]['ir_frames']['timestamps'].append(ir_frame.get_timestamp())
                        self.result_dataset[cam_name]['depth_frames']['copy'].append(depth_frame)
                        self.result_dataset[cam_name]['depth_frames']['timestamps'].append(depth_frame.get_timestamp())

                elif self.policy == 'drop':
                    frames = pipeline.wait_for_frames(100)
                    frame_index += 1
                    if frame_index >= last_frame:
                        break
                    elif frames is None:
                        print(f'\n[ERROR]: Frameset Lost for Frame index: {frame_index} {cam_name}\n')
                        self.drop_list.append(frame_index)
                        self.replace_none(cam_name) # We add ' None ' value because first we should keep the cardinality of the Result Data set same for all cameras.
                        self.result_dataset[cam_name]['ir_frames']['abnormal_tms_index'].append(frame_index)
                        self.result_dataset[cam_name]['depth_frames']['abnormal_tms_index'].append(frame_index)
                    elif frames.get_depth_frame() is None or frames.get_ir_frame() is None:
                        print(f'\n[ERROR]: IR or Depth Frame Lost for Frame index: {frame_index} {cam_name}\n')
                        self.drop_list.append(frame_index)
                        self.replace_none(cam_name)
                        self.result_dataset[cam_name]['ir_frames']['abnormal_tms_index'].append(frame_index)
                        self.result_dataset[cam_name]['depth_frames']['abnormal_tms_index'].append(frame_index)
                    else:
                        depth_frame = frames.get_depth_frame()
                        ir_frame = frames.get_ir_frame()
                        self.result_dataset[cam_name]['ir_frames']['drop'].append(ir_frame)
                        self.result_dataset[cam_name]['ir_frames']['timestamps'].append(ir_frame.get_timestamp())
                        self.result_dataset[cam_name]['depth_frames']['drop'].append(depth_frame)
                        self.result_dataset[cam_name]['depth_frames']['timestamps'].append(depth_frame.get_timestamp())
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            pipeline.stop()
            print('\n[INFO] Data extraction completed')

    def replace_none(self, cam_name):
        """Replace 'None' value for abnormal frames."""
        self.result_dataset[cam_name]['ir_frames']['drop'].append(None)
        self.result_dataset[cam_name]['ir_frames']['timestamps'].append(None)
        self.result_dataset[cam_name]['depth_frames']['drop'].append(None)
        self.result_dataset[cam_name]['depth_frames']['timestamps'].append(None)

    def tmscheck(self, policy):
        """Check timestamps for reliability."""
        for camera_name, data in self.result_dataset.items():
            for idx, tms in enumerate(data['ir_frames']['timestamps']):
                if policy == 'copy':
                    if (idx > 0) and not (idx in data['ir_frames']['abnormal_tms_index']) and not (idx - 1 in data['ir_frames']['abnormal_tms_index']):
                        if abs(tms - data['ir_frames']['timestamps'][idx - 1]) > 135:
                            print(f'\n[ERROR]: Unusual timestamp for camera: {camera_name} Frame index: {idx}\n')
                            self.replacment_frmset(idx, camera_name, None, True)
                elif policy == 'drop':
                    if idx > 0 and not (idx in data['ir_frames']['abnormal_tms_index']) and not (idx - 1 in data['ir_frames']['abnormal_tms_index']):
                        if abs(tms - data['ir_frames']['timestamps'][idx - 1]) > 135:
                            print(f'\n[ERROR]: Unusual timestamp for camera: {camera_name} Frame index: {idx}\n')
                            self.drop_list.append(idx)
            for idx, tms in enumerate(data['depth_frames']['timestamps']):
                if policy == 'copy':
                    if (idx > 0) and not (idx in data['depth_frames']['abnormal_tms_index']) and not (idx - 1 in data['depth_frames']['abnormal_tms_index']):
                        if abs(tms - data['depth_frames']['timestamps'][idx - 1]) > 135:
                            print(f'\n[ERROR]: Unusual timestamp for camera: {camera_name} Frame index: {idx}\n')
                            self.replacment_frmset(idx, camera_name, None, True)
                elif policy == 'drop':
                    if idx > 0 and not (idx in data['depth_frames']['abnormal_tms_index']) and not (idx - 1 in data['depth_frames']['abnormal_tms_index']):
                        if abs(tms - data['depth_frames']['timestamps'][idx - 1]) > 135:
                            print(f'\n[ERROR]: Unusual timestamp for camera: {camera_name} Frame index: {idx}\n')
                            self.drop_list.append(idx)

    def cardinality_check(self, policy):
        """Check the cardinality of the framesets."""
        first_cam_name = next(iter(self.result_dataset))
        first_cam_cardi = np.shape(self.result_dataset[first_cam_name]['ir_frames'][policy])
        for camera_name, data in self.result_dataset.items():
            if np.shape(data['ir_frames'][policy]) != first_cam_cardi or np.shape(data['depth_frames'][policy]) != first_cam_cardi:
                print(f'\n[ERROR] The cardinality of frames set of the camera: {camera_name} is abnormal')
                return False
            elif np.shape(data['ir_frames']['timestamps']) != first_cam_cardi or np.shape(data['depth_frames']['timestamps']) != first_cam_cardi:
                print(f'\n[ERROR] The cardinality of timestamps of the camera: {camera_name} is abnormal')
                return False
        return True

    def replacment_frmset(self, frame_index, camera_name, replace_tms=False):
        """Replace frameset with abnormal timestamps."""
        if replace_tms:
            replacement_tms = self.extract_replacment_tms(frame_index, camera_name)
            try:
                self.result_dataset[camera_name]['depth_frames']['timestamps'][frame_index] = replacement_tms
                self.result_dataset[camera_name]['ir_frames']['timestamps'][frame_index] = replacement_tms
            except IndexError:
                self.result_dataset[camera_name]['depth_frames']['timestamps'].append(replacement_tms)
                self.result_dataset[camera_name]['ir_frames']['timestamps'].append(replacement_tms)

        replacement_irframe = self.result_dataset[camera_name]['ir_frames']['copy'][frame_index - 1]
        replacement_depthframe = self.result_dataset[camera_name]['depth_frames']['copy'][frame_index - 1]
        if replacement_irframe and replacement_depthframe:
            try:
                self.result_dataset[camera_name]['depth_frames']['copy'][frame_index] = replacement_depthframe
                self.result_dataset[camera_name]['ir_frames']['copy'][frame_index] = replacement_irframe
            except IndexError:
                self.result_dataset[camera_name]['depth_frames']['copy'].append(replacement_depthframe)
                self.result_dataset[camera_name]['ir_frames']['copy'].append(replacement_irframe)

    def extract_replacment_tms(self, frm_indx, cam_name):
        """Extract replacement timestamp for frameset."""
        for camera_name, datasets in self.result_dataset.items():
            tms = datasets['ir_frames']['timestamps'][frm_indx]
            if tms is not None and camera_name != cam_name:
                if frm_indx not in datasets['ir_frames']['abnormal_tms_index']:
                    return tms
        prev_tms = self.result_dataset[cam_name]['ir_frames']['timestamps'][frm_indx - 1]
        self.result_dataset[cam_name]['ir_frames']['abnormal_tms_index'].append(frm_indx)
        return prev_tms

    def drop_frmsets(self):
        """Drop framesets marked for removal."""
        self.drop_list = list(set(self.drop_list))
        self.drop_list.sort(reverse=True)
        for camera_name, frameset in self.result_dataset.items():
            for dataset, sets in frameset.items():
                for index in self.drop_list:
                    try:
                        sets['drop'].pop(index)
                        sets['timestamps'].pop(index)
                        sets['removed_frms'].append(index)
                    except IndexError:
                        continue

    def save_tms_file(self):
        """Save timestamp information to file."""
        for camera_name, data in self.result_dataset.items():
            for dataset, set in data.items():
                timestamp_foldername = os.path.join(self.output_path, camera_name, dataset)
                self.create_output_dir(timestamp_foldername)
                timestamp_filename = os.path.join(timestamp_foldername, 'frames_timestamp.txt')
                tms_file = np.array(set['timestamps'])
                with open(timestamp_filename, 'a') as f:
                    for tms in tms_file:
                        f.write(str(tms) + '\n')
                print(f"\n[INFO] The Timestamp file for {camera_name} {dataset} saved at: {timestamp_filename}")

    def create_output_dir(self, output_path: str):
        """Create the output directory if it doesn't exist."""
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def get_device_info(self, output_dir, cam_name, device_info, camera_param):
        """Save device information to file."""
        folder_path = os.path.join(output_dir, cam_name)
        self.create_output_dir(folder_path)
        deviceinfo_path = os.path.join(folder_path, 'device_info.txt')
        with open(deviceinfo_path, 'w') as deviceinfo_file:
            deviceinfo_file.write(f'\nCamera Name: {cam_name}\n')
            deviceinfo_file.write(f'\nCamera Parameters: {camera_param}\n')
            deviceinfo_file.write(f'\n{device_info}\n')

    def generate_metadata(self):
        """Generate metadata for each frameset."""
        for camera_name, data in self.result_dataset.items():
            for dataset, frameset in data.items():
                metadata_filepath = os.path.join(self.output_path, camera_name, dataset, f"{dataset}_metadata.txt")
                with open(metadata_filepath, 'w') as metadata_file:
                    frame = frameset[self.policy][0]
                    metadata_file.write(f'\nFrame Index: {frame.get_index()}\n')
                    metadata_file.write(f'Data Size (byte): {frame.get_data_size()}\n')
                    metadata_file.write(f'Format: {frame.get_format()}\n')
                    metadata_file.write(f'Height: {frame.get_height()}\n')
                    metadata_file.write(f'Width: {frame.get_width()}\n')
                    metadata_file.write(f'Pixel Bit Size: {frame.get_pixel_available_bit_size()}\n')
                    metadata_file.write(f'Type: {frame.get_type()}\n')

    def save_depth_frame(self, start_frame):
        """Save depth frames as image files."""
        for camera_name, data in self.result_dataset.items():
            for dataset, frameset in data.items():
                if dataset == 'depth_frames':
                    frames_foldername = os.path.join(self.output_path, camera_name, dataset, 'png')
                    self.create_output_dir(frames_foldername)
                    frame_index = start_frame
                    for frame in frameset[self.policy]:
                        data = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))
                        data = (data * frame.get_depth_scale()).astype(np.uint16)
                        if self.convert_depth_to_8bit:
                            normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
                            depth_image = cv2.applyColorMap(normalized_data.astype(np.uint8), cv2.COLORMAP_JET)
                            filename = os.path.join(frames_foldername, f"depth_{frame.get_width()}x{frame.get_height()}_{frame_index}_8bit.png")
                        else:
                            depth_image = data
                            filename = os.path.join(frames_foldername, f"depth_{frame.get_width()}x{frame.get_height()}_{frame_index}_16bit.png")
                        frame_index += 1
                        cv2.imwrite(filename, depth_image)
                    print(f"\n[INFO] Depth Frames for {camera_name} saved at: {frames_foldername}\n")

    def save_ir_frame(self, start_frame):
        """Save IR frames as image files."""
        for camera_name, data in self.result_dataset.items():
            for dataset, frameset in data.items():
                if dataset == 'ir_frames':
                    frames_foldername = os.path.join(self.output_path, camera_name, dataset, 'png')
                    self.create_output_dir(frames_foldername)
                    frame_index = start_frame
                    for frame in frameset[self.policy]:
                        data = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((frame.get_height(), frame.get_width()))
                        if self.convert_ir_to_8bit:
                            normalized_data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
                            ir_image = normalized_data.astype(np.uint8)
                            filename = os.path.join(frames_foldername, f"ir_{frame.get_width()}x{frame.get_height()}_{frame_index}_8bit.png")
                        else:
                            ir_image = data
                            filename = os.path.join(frames_foldername, f"ir_{frame.get_width()}x{frame.get_height()}_{frame_index}_16bit.png")
                        frame_index += 1
                        cv2.imwrite(filename, ir_image)
                    print(f"\n[INFO] IR Frames for {camera_name} saved at: {frames_foldername}")
