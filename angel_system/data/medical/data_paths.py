import os
from glob import glob

KNOWN_BAD_VIDEOS = ["M2-15"]  # Videos without any usable data

TASK_TO_NAME = {
    'm1': "M1_Trauma_Assessment",
    'm2': "M2_Tourniquet",
    'm3': "M3_Pressure_Dressing",
    'm4': "M4_Wound_Packing",
    'm5': "M5_X-Stat",
    'r18': "R18_Chest_Seal",
}

LAB_TASK_TO_NAME = {
    'm1': "M2_Lab_Skills",
    'm2': "M2_Lab_Skills",
    'm3': "M3_Lab_Skills",
    'm4': "M4_Wound_Packing",
    'm5': "M5_Lab_Skills",
    'r18': "R18_Lab_Skills",
}


def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types from directories, optionally recursively.

    Parameters:
        path (str): Root directory path.
        types (list): List of file types (extensions) to be extracted.
        recursive (bool, optional): Search for files in subsequent directories if True. Default is False.

    Returns:
        list: List of file paths with full paths.
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files

class GrabData(object):
    def __init__(self, yaml_path: str):
        
        import yaml
        with open(yaml_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.data_dir = config['data_dir']
        self.activity_gt_dir = config['activity_gt_dir']
        self.objects_dir = config['objects_dir']
        self.ros_bags_dir = config['ros_bags_dir']
        
        self.bbn_data_dir = config['bbn_data_dir']
        self.bbn_data_root = config['bbn_data_root']
        
        self.lab_bbn_data_root = config['lab_bbn_data_root']
    
    def grab_data(self, skill: str, data_type: str) -> list:
        if data_type == "pro":
            skill_data_root = f"{self.bbn_data_root}/{TASK_TO_NAME[skill]}/Data"
            videos = os.listdir(skill_data_root)
            
            videos_paths = [f"{skill_data_root}/{video}" for video in videos]
            
            return videos_paths
            
        elif data_type == 'lab':
            skill_data_root = f"{self.lab_bbn_data_root}/{LAB_TASK_TO_NAME[skill]}/"
            
            videos_paths = dictionary_contents(self.skill_data_root, types=['*.mp4'])
            
            return videos_paths
        
        else:
            raise NotImplementedError

