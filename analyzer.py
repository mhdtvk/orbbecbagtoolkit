import os
import numpy as np
import matplotlib.pyplot as plt

# Use this script to analyze the timestamps of the frames in the acquisition folders extracted from the camera bags (after run_datagen.sh)

# read timestamps from file
def read_timestamps(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [int(line.strip()) for line in lines if line.strip()]

# compute deltas between timestamps
def calculate_deltas(timestamps):
    return np.diff(timestamps)

# count lost frames
def count_lost_frames(deltas, threshold=100):
    return np.sum(deltas > threshold)

# analyuze acquisition folder
def analyze_acquisition_folder(acquisition_folder):
    results = {}
    cameras = ['camera_0', 'camera_1', 'camera_2'] # aumenta il numero di camere se necessario
    image_types = ['ir_frames', 'depth_frames']
    for camera in cameras:
        results[camera] = {}
        for image_type in image_types:
            file_path = os.path.join(acquisition_folder, camera, image_type, "frames_timestamp.txt")
            timestamps = read_timestamps(file_path)
            deltas = calculate_deltas(timestamps)
            lost_frames = count_lost_frames(deltas)
            results[camera][image_type] = {
                'timestamps': timestamps,
                'deltas': deltas,
                'lost_frames': lost_frames
            }
    return results

# plot deltas
def plot_deltas(results, stats_folder):
    cameras = ['camera_0', 'camera_1', 'camera_2']
    image_types = ['depth_frames', 'ir_frames']
    for camera in cameras:
        for image_type in image_types:
            deltas = results[camera][image_type]['deltas']
            plt.figure()
            plt.plot(deltas)
            plt.title(f'Deltas for {camera} {image_type}')
            plt.xlabel('Frame index')
            plt.ylabel('Delta (ms)')
            plt.savefig(os.path.join(stats_folder, f'{camera}_{image_type}_deltas.png'))
            plt.close()

# intercamera analysis
def analyze_intercamera(results):
    intercamera_results = {}
    cameras = ['camera_0', 'camera_1', 'camera_2']
    image_types = ['depth_frames', 'ir_frames']
    for image_type in image_types:
        for i in range(len(cameras)):
            for j in range(i+1, len(cameras)):
                cam1 = cameras[i]
                cam2 = cameras[j]
                timestamps1 = results[cam1][image_type]['timestamps']
                timestamps2 = results[cam2][image_type]['timestamps']
                
                # Interpolazione lineare per gestire diverse lunghezze
                min_len = min(len(timestamps1), len(timestamps2))
                timestamps1 = np.array(timestamps1[:min_len])
                timestamps2 = np.array(timestamps2[:min_len])
                
                deltas = np.abs(timestamps1 - timestamps2)
                lost_frames = count_lost_frames(deltas)
                pair_key = f'{cam1}_vs_{cam2}'
                if pair_key not in intercamera_results:
                    intercamera_results[pair_key] = {}
                intercamera_results[pair_key][image_type] = {
                    'deltas': deltas,
                    'lost_frames': lost_frames
                }
    return intercamera_results

# plot intercamera deltas results
def plot_intercamera_deltas(intercamera_results, stats_folder):
    for pair_key, image_type_results in intercamera_results.items():
        for image_type, data in image_type_results.items():
            deltas = data['deltas']
            plt.figure()
            plt.plot(deltas)
            plt.title(f'Intercamera Deltas for {pair_key} {image_type}')
            plt.xlabel('Frame index')
            plt.ylabel('Delta (ms)')
            plt.savefig(os.path.join(stats_folder, f'{pair_key}_{image_type}_deltas.png'))
            plt.close()

# main function
def main(acquisition_folder):
    stats_folder = os.path.join(acquisition_folder, 'stats')
    os.makedirs(stats_folder, exist_ok=True)
    
    results = analyze_acquisition_folder(acquisition_folder)
    plot_deltas(results, stats_folder)
    intercamera_results = analyze_intercamera(results) 
    plot_intercamera_deltas(intercamera_results, stats_folder)

    # Stampa dei risultati
    for camera, image_types in results.items():
        for image_type, data in image_types.items():
            print(f'{camera} {image_type}: Lost frames = {data["lost_frames"]}')

    for pair_key, image_type_results in intercamera_results.items():
        for image_type, data in image_type_results.items():
            print(f'{pair_key} {image_type}: Lost frames = {data["lost_frames"]}')



# run main for each acquisition folder
folder1 = os.path.join(os.getcwd(), "ir_depth_raw_files/raw_data_to_test_8bit_exp/calibrazione_camera_statica_20240625131330/")
enter_new_path = input(f"\nThe default path is : {folder1} \nTo enter a new one press '1' , To 'skip' press 'Enter'\n")
if enter_new_path == '1' : 
    folder1 = input("\nEnter the new Path of the acquisition folder :")
folder_list = [folder1]
for acquisition_folder in folder_list:
    print(f'Analyzing {acquisition_folder}')
    main(acquisition_folder)
    print('-----------------------------------')
